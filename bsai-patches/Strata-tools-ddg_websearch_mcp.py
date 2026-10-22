#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddg_websearch_mcp.py — a stdio MCP server giving the Strata model a real web search.

Why this exists
---------------
Strata offers tools only from MCP servers (`mcp_servers` in the run config, or
`--mcp-config`).  With none configured, `GET /mcp` reports
`{"servers": [], "tools": 0}` and the model has nothing to call -- so when the
user asks for a web search it narrates one ("正在联网搜索: ...") and stops.  That
is the model hallucinating a tool, not a bug in it.

Transport: stdio, newline-delimited JSON-RPC 2.0.  STDOUT carries the protocol
and nothing else -- every diagnostic goes to stderr, because a stray print()
there corrupts the stream.

Search backend: Bing.  DuckDuckGo / Google / Brave are all unreachable from a
mainland-China network (verified: every one of them times out), so Bing is the
backend that actually answers here.  No API key, standard library only.

Methods answered: initialize, notifications/initialized, ping, tools/list,
tools/call.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from html.parser import HTMLParser

PROTOCOL = "2024-11-05"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT = 12
BING_HOSTS = ("cn.bing.com", "www.bing.com")

# 同一查询 90 秒内重复调用直接返回缓存，避免小模型陷入工具循环
# （实测 IQ3_S 会把同一个 "免费在线游戏" 连搜 4 次、把上轮结果里的 URL 当新 query）
_SEARCH_CACHE = {}
DEDUP_TTL = 90


def log(*a):
    """stderr only - stdout is the JSON-RPC channel."""
    try:
        print(*a, file=sys.stderr, flush=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- search
class _BingParser(HTMLParser):
    """Pull title / url / snippet out of Bing's results page.

    Each hit is  <li class="b_algo"> <h2><a href=...>Title</a></h2>
                 <div class="b_caption"> ...snippet... </div> </li>
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._in_algo = False
        self._in_h2 = False
        self._in_cap = False
        self._pending_href = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "li" and "b_algo" in cls.split():
            self._in_algo = True
            self._pending_href = None
        elif tag == "h2" and self._in_algo:
            self._in_h2 = True
        elif tag == "a" and self._in_h2:
            self._pending_href = a.get("href")
            self._buf = []
        elif tag == "div" and "b_caption" in cls.split() and self._in_algo:
            self._in_cap = True
        elif tag in ("p", "div") and self._in_cap:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "a" and self._in_h2:
            title = " ".join("".join(self._buf).split())
            if title and self._pending_href:
                self.results.append({"title": title, "url": self._pending_href, "snippet": ""})
            self._in_h2 = False
            self._pending_href = None
        elif tag == "h2":
            self._in_h2 = False
        elif tag == "div" and self._in_cap:
            if self.results and self._in_algo:
                snip = " ".join("".join(self._buf).split())
                if snip:
                    cur = self.results[-1]
                    cur["snippet"] = (cur["snippet"] + " " + snip).strip()[:600]
            self._in_cap = False
            self._buf = []
        elif tag == "li" and self._in_algo:
            self._in_algo = False
            self._in_cap = False
            self._buf = []

    def handle_data(self, data):
        if self._in_h2 or self._in_cap:
            self._buf.append(data)


def _fetch(url, data=None):
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def _clean_query(q):
    """净化模型传来的 query：剥引号、截断、识别误传的 URL。"""
    q = str(q or "").strip()
    if not q:
        return ""
    if q.startswith(("http://", "https://")):
        return q                                    # 调用方转 fetch_url
    q = q.strip("'\"“”‘’`")
    if len(q) > 100:
        q = q[:100].rstrip()
    return q


def web_search(query, count=6):
    """Bing, no API key. Tries cn.bing.com then www.bing.com."""
    query = str(query).strip()
    if not query:
        return []
    count = max(1, min(int(count or 6), 15))
    qs = urllib.parse.urlencode({"q": query, "setlang": "zh-CN", "count": str(count + 4)})
    last_err = None
    for host in BING_HOSTS:
        try:
            html = _fetch("https://%s/search?%s" % (host, qs))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
            log("[websearch] %s failed: %s" % (host, e))
            continue
        p = _BingParser()
        try:
            p.feed(html)
        except Exception as e:
            log("[websearch] parse error: %s" % e)
        out, seen = [], set()
        for r in p.results:
            u = r["url"]
            if u in seen or not u.startswith("http"):
                continue
            seen.add(u)
            out.append(r)
            if len(out) >= count:
                break
        if out:
            return out
        last_err = "no results parsed"
    raise RuntimeError("search failed: %s" % (last_err or "unknown"))


def fetch_url(url, max_chars=6000):
    """Read one page and hand back its visible text (tool results stay small)."""
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        return "only http(s) urls are supported"
    raw = _fetch(url)
    txt = " ".join(_tags_to_ws(_strip_blocks(raw)).split())
    if len(txt) > int(max_chars or 6000):
        txt = txt[:int(max_chars)] + " ...[truncated]"
    return txt or "(page had no readable text)"


def _strip_blocks(html):
    low, out, i = html.lower(), [], 0
    for tag in ("script", "style", "noscript"):
        while True:
            s = low.find("<" + tag, i)
            if s < 0:
                break
            e = low.find("</" + tag, s)
            if e < 0:
                i = len(html)
                break
            e = html.find(">", e) + 1
            out.append(html[i:s])
            i = e
    out.append(html[i:])
    return "".join(out)


def _tags_to_ws(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "<":
            j = s.find(">", i)
            if j < 0:
                break
            out.append(" ")
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------- tools
TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the public web and return ranked results with title, URL and snippet. "
            "Use it whenever the answer depends on current events, recent releases, "
            "prices, news, or anything after your knowledge cutoff. "
            "For questions about the user's own files or local system, use other means - "
            "this searches the internet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "2-6 short keywords, e.g. 'Proton VPN free 2026'. Not a full sentence, not a URL. Never repeat the same query twice."},
                "count": {"type": "integer", "description": "How many results (1-15, default 6)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a web page and return its readable text. Use after web_search to read a promising result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL."},
                "max_chars": {"type": "integer", "description": "Truncate the text here (default 6000)."},
            },
            "required": ["url"],
        },
    },
]


def call_tool(name, args):
    if name == "web_search":
        query = _clean_query(args.get("query", ""))
        if not query:
            return "No query given."
        if query.startswith(("http://", "https://")):
            return fetch_url(query)
        now = time.time()
        hit = _SEARCH_CACHE.get(query)
        if hit and now - hit[1] < DEDUP_TTL:
            return hit[0] + (
                "\n\n[note: this exact query was already searched %ds ago - "
                "do not call web_search again with the same words; refine the keywords, "
                "or fetch a result URL with fetch_url.]" % int(now - hit[1]))
        res = web_search(query, args.get("count", 6))
        if not res:
            return "No results found for %r." % query[:120]
        text = "\n\n".join(
            "%d. %s\n   %s\n   %s" % (i, r["title"], r["url"], r.get("snippet") or "")
            for i, r in enumerate(res, 1))
        _SEARCH_CACHE[query] = (text, now)
        return text
    if name == "fetch_url":
        try:
            return fetch_url(args.get("url", ""), args.get("max_chars", 6000))
        except Exception as e:
            return ("Cannot open %s (%s). It timed out or refused the connection - "
                    "do NOT retry this exact URL; pick a different result from the search list, "
                    "or answer from the snippets you already have."
                    % (str(args.get("url", ""))[:120], type(e).__name__))
    raise ValueError("unknown tool: %s" % name)


# --------------------------------------------------------------------------- JSON-RPC loop
def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def ok(req_id, result):
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def err(req_id, code, message):
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def handle(msg):
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return ok(req_id, {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "bing-websearch", "title": "Bing Web Search", "version": "1.0"},
        })
    if method in ("notifications/initialized", "initialized"):
        return None                                    # notification: no reply
    if method == "ping":
        return ok(req_id, {})
    if method == "tools/list":
        return ok(req_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text = call_tool(name, args)
            return ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:                          # tool errors are results, not protocol errors
            log("[websearch] tool %s failed: %s" % (name, e))
            return ok(req_id, {"content": [{"type": "text", "text": "tool error: %s" % e}],
                               "isError": True})
    if req_id is not None:
        return err(req_id, -32601, "method not found: %s" % method)
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    log("[websearch] ready on stdio (Bing backend, no API key)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            log("[websearch] bad json line ignored")
            continue
        if not isinstance(msg, dict):
            continue
        batch = msg if isinstance(msg, list) else [msg]     # a client may batch
        for one in batch:
            if isinstance(one, dict):
                try:
                    handle(one)
                except Exception as e:
                    log("[websearch] handler error: %s" % e)
                    if one.get("id") is not None:
                        err(one["id"], -32603, "internal error: %s" % e)


if __name__ == "__main__":
    main()
