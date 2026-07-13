from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
POSTS = CONTENT / "posts"
TEMPLATES = ROOT / "templates"


@dataclass
class Document:
    meta: dict[str, str]
    body: str
    source: Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def parse_document(path: Path) -> Document:
    text = read_text(path)
    meta: dict[str, str] = {}
    body = text

    if text.startswith("---\n"):
        _, raw_meta, body = text.split("---\n", 2)
        for line in raw_meta.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    return Document(meta=meta, body=body.strip(), source=path)


def slug_from_post(path: Path) -> str:
    return path.with_suffix(".html").name


def markdown_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    code_block: list[str] | None = None
    code_language = ""

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(line.strip() for line in paragraph)
            blocks.append(f"<p>{markdown_inline(text)}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            body = "\n".join(f"        <li>{item}</li>" for item in list_items)
            blocks.append(f"<ul>\n{body}\n    </ul>")
            list_items = []

    def flush_code_block() -> None:
        nonlocal code_block, code_language
        if code_block is not None:
            code = "\n".join(html.escape(line) for line in code_block)
            language = html.escape(code_language)
            class_attr = f' class="language-{language}"' if language else ""
            blocks.append(f"<pre><code{class_attr}>{code}</code></pre>")
            code_block = None
            code_language = ""

    for line in lines:
        stripped = line.strip()
        if code_block is not None:
            if stripped.startswith("```"):
                flush_code_block()
            else:
                code_block.append(line.rstrip())
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            code_language = stripped[3:].strip()
            code_block = []
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        image_match = re.fullmatch(r'!\[([^\]]*)\]\((\S+)(?:\s+"([^"]+)")?\)', stripped)
        if image_match:
            flush_paragraph()
            flush_list()
            alt, src, caption = image_match.groups()
            alt = html.escape(alt)
            src = html.escape(src, quote=True)
            caption_html = f"\n    <figcaption>{markdown_inline(caption)}</figcaption>" if caption else ""
            blocks.append(f'<figure class="post-figure">\n    <img src="{src}" alt="{alt}" loading="lazy">{caption_html}\n</figure>')
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            flush_list()
            blocks.append(f"<h3>{markdown_inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            blocks.append(f"<h2>{markdown_inline(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            flush_paragraph()
            list_items.append(markdown_inline(stripped[2:]))
        else:
            flush_list()
            paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    flush_code_block()
    return "\n".join(blocks)


def render_template(name: str, context: dict[str, str]) -> str:
    output = read_text(TEMPLATES / name)
    for key, value in context.items():
        output = output.replace("{{ " + key + " }}", value)
    return output


def wrap_base(site: dict[str, str], title: str, subtitle: str, content: str) -> str:
    page_title = site["title"] if title == site["title"] else f"{title} - {site['title']}"
    return render_template(
        "base.html",
        {
            "page_title": html.escape(page_title),
            "description": html.escape(site["description"]),
            "title": html.escape(title),
            "subtitle": html.escape(subtitle),
            "content": content,
            "year": html.escape(site["year"]),
            "author": html.escape(site["author"]),
        },
    )


def section_cards(markdown: str) -> str:
    raw = markdown_to_html(markdown)
    chunks = re.split(r"(?=<h2>)", raw)
    sections = []
    for chunk in chunks:
        if chunk.strip():
            sections.append(f'<section class="card">\n{chunk.strip()}\n</section>')
    return "\n\n".join(sections)


def load_posts() -> list[dict[str, str]]:
    posts: list[dict[str, str]] = []
    for path in sorted(POSTS.glob("*/*.md")):
        doc = parse_document(path)
        category = doc.meta["category"]
        slug = slug_from_post(path)
        posts.append(
            {
                **doc.meta,
                "body": markdown_to_html(doc.body),
                "url": f"/posts/{category}/{slug}",
                "output": f"posts/{category}/{slug}",
            }
        )
    return sorted(posts, key=lambda item: item["date"], reverse=True)


def post_item(post: dict[str, str]) -> str:
    summary = post.get("summary", "")
    summary_html = f'<p class="article-summary">{html.escape(summary)}</p>' if summary else ""
    return (
        "        <li>\n"
        f'            <a href="{post["url"]}">{html.escape(post["title"])}</a>\n'
        f'            <span class="article-date">{html.escape(post["date"])}</span>\n'
        f"            {summary_html}\n"
        "        </li>"
    )


def post_routes(post: dict[str, str]) -> set[str]:
    raw = post.get("routes", post.get("route", ""))
    return {item.strip() for item in raw.split(",") if item.strip()}


def simple_intro_section(heading: str, items: list[str]) -> str:
    intro_items = "\n".join(f"        <li>{html.escape(item)}</li>" for item in items)
    return (
        '<section class="card">\n'
        f"    <h2>{html.escape(heading)}</h2>\n"
        "    <ul>\n"
        f"{intro_items}\n"
        "    </ul>\n"
        "</section>"
    )


def route_post_items(posts: list[dict[str, str]]) -> str:
    if not posts:
        return '            <li class="route-empty">暂无文章</li>'
    return "\n".join(
        "            <li>\n"
        f'                <a href="{post["url"]}">{html.escape(post["title"])}</a>\n'
        f'                <span class="article-date">{html.escape(post["date"])}</span>\n'
        "            </li>"
        for post in posts
    )


def route_branch_item(branch: dict[str, object], posts: list[dict[str, str]]) -> str:
    items = "\n".join(f"            <li>{html.escape(item)}</li>" for item in branch["items"])
    key = str(branch["key"])
    related_posts = [post for post in posts if key in post_routes(post)]
    post_items = route_post_items(related_posts)
    return (
        '    <article class="route-branch">\n'
        f"        <h3>{html.escape(str(branch['title']))}</h3>\n"
        f"        <p>{html.escape(str(branch['summary']))}</p>\n"
        "        <ul>\n"
        f"{items}\n"
        "        </ul>\n"
        '        <div class="route-posts">\n'
        "            <h4>相关文章</h4>\n"
        "            <ul>\n"
        f"{post_items}\n"
        "            </ul>\n"
        "        </div>\n"
        "    </article>"
    )


def route_intro_section(data: dict[str, object], posts: list[dict[str, str]]) -> str:
    branches = "\n".join(route_branch_item(branch, posts) for branch in data["branches"])
    return (
        '<section class="route-section">\n'
        f"    <h2>{html.escape(str(data['heading']))}</h2>\n"
        f"    <p class=\"route-intro\">{html.escape(str(data['intro']))}</p>\n"
        '    <div class="route-grid">\n'
        f"{branches}\n"
        "    </div>\n"
        "</section>"
    )


def build_posts(site: dict[str, str], posts: list[dict[str, str]]) -> None:
    for post in posts:
        content = render_template(
            "post.html",
            {
                "date": html.escape(post["date"]),
                "category": html.escape(post["category"]),
                "category_name": html.escape(post["category_name"]),
                "body": post["body"],
            },
        )
        page = wrap_base(site, post["title"], f"发布时间：{post['date']}", content)
        write_text(ROOT / post["output"], page)


def build_lists(site: dict[str, str], posts: list[dict[str, str]]) -> None:
    categories = {
        "tech": {
            "title": "技术笔记",
            "subtitle": "记录密码工程、GPU 高性能计算、AI 算子库实现与工程管理实践。",
            "heading": "技术路线",
            "intro": "这里按长期积累方向组织技术内容。每条分支既可以沉淀单点技术，也可以逐步串成项目复盘、工程实践和方法论。",
            "branches": [
                {
                    "key": "crypto-engineering",
                    "title": "密码工程",
                    "summary": "关注密码算法从论文、标准到工程实现的落地过程。",
                    "items": ["后量子密码学实现", "隐私计算与安全协议", "正确性测试与性能基准", "常数时间实现与安全边界"],
                },
                {
                    "key": "gpu-hpc",
                    "title": "GPU 高性能计算",
                    "summary": "围绕 CUDA 编程、性能分析和硬件友好计算组织展开。",
                    "items": ["CUDA kernel 编写与调优", "访存合并、共享内存与并行粒度", "Nsight profiling 与瓶颈分析", "HPC 思路在 AI 推理中的迁移"],
                },
                {
                    "key": "operator-library",
                    "title": "算子库实现",
                    "summary": "面向推理系统中的 AI 算子，记录 kernel 实现、算子库入口、测试和性能基准。",
                    "items": [
                        "add / GEMM / softmax / layernorm 等 AI 算子实现",
                        "Triton kernel 编写、注册与算子库集成",
                        "算子融合、量化、attention 和 KV cache 相关优化",
                        "跨后端性能对比、benchmark 与回归验证",
                    ],
                },
                {
                    "key": "team-collaboration",
                    "title": "团队管理与工程协作",
                    "summary": "沉淀项目推进中的协作方式、GitHub 团队流程、质量保障和技术决策过程。",
                    "items": [
                        "任务拆解、里程碑管理与进度同步",
                        "GitHub issue / branch / pull request 协作流程",
                        "Code Review、工程规范与文档沉淀",
                        "CI 检查、回归测试、风险识别和阶段复盘",
                    ],
                },
            ],
        },
        "life": {
            "title": "生活随笔",
            "subtitle": "记录读书、日常心得、阶段复盘与个人成长。",
            "heading": "内容方向",
            "items": ["读书笔记", "日常思考", "阶段总结", "学习与成长", "求职过程中的复盘"],
        },
    }

    for category, data in categories.items():
        category_posts = [post for post in posts if post["category"] == category]
        intro_section = route_intro_section(data, category_posts) if "branches" in data else simple_intro_section(data["heading"], data["items"])
        content = render_template(
            "list.html",
            {
                "intro_section": intro_section,
                "post_items": "\n".join(post_item(post) for post in category_posts),
            },
        )
        page = wrap_base(site, data["title"], data["subtitle"], content)
        write_text(ROOT / f"{category}.html", page)


def build_pages(site: dict[str, str], posts: list[dict[str, str]]) -> None:
    for path in sorted((CONTENT / "pages").glob("*.md")):
        doc = parse_document(path)
        sections = section_cards(doc.body)
        template = doc.meta.get("template", "page.html")
        output = doc.meta.get("output", "index.html" if path.stem == "home" else f"{path.stem}.html")
        content = render_template(
            template,
            {
                "sections": sections,
                "featured_posts": "\n".join(post_item(post) for post in posts[:5]),
            },
        )
        page = wrap_base(site, doc.meta["title"], doc.meta.get("subtitle", site["subtitle"]), content)
        write_text(ROOT / output, page)


def main() -> None:
    site = json.loads(read_text(ROOT / "site.json"))
    posts = load_posts()
    build_posts(site, posts)
    build_lists(site, posts)
    build_pages(site, posts)
    print(f"Built {len(posts)} posts and 4 top-level pages.")


if __name__ == "__main__":
    main()
