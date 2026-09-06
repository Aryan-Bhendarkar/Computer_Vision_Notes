#!/usr/bin/env python3
"""Inline content.json into template.html -> index.html"""
import pathlib, sys
t = pathlib.Path("template.html").read_text(encoding="utf-8")
c = pathlib.Path("content.json").read_text(encoding="utf-8")
c = c.replace("</script>", "<\\/script>").replace("<!--", "<\\!--")
out = t.replace("__CONTENT__", c)
pathlib.Path("index.html").write_text(out, encoding="utf-8")
print(f"index.html  {len(out)/1e6:.2f} MB")
