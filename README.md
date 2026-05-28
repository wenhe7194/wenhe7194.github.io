# Wenhe 的博客

这是一个零依赖 Python 静态博客生成器项目。内容写在 `content/`，页面模板写在 `templates/`，生成结果输出为 GitHub Pages 可直接托管的 HTML 文件。

## 生成站点

```powershell
python scripts/build.py
```

## 目录说明

- `site.json`：站点标题、作者、描述等全局配置。
- `content/pages/`：首页、关于页等独立页面内容。
- `content/posts/tech/`：技术笔记 Markdown 源文件。
- `content/posts/life/`：生活随笔 Markdown 源文件。
- `templates/`：自定义 HTML 模板。
- `assets/css/style.css`：全站样式。
- `index.html`、`tech.html`、`life.html`、`about.html`、`posts/**/*.html`：生成结果，用于 GitHub Pages 部署。

## 新增文章

在对应分类目录新建 Markdown 文件，例如：

```text
content/posts/tech/2026-05-28-example.md
```

文件头部使用：

```markdown
---
title: 文章标题
date: 2026-05-28
category: tech
category_name: 技术笔记
summary: 文章摘要
---
```

保存后运行 `python scripts/build.py`。
