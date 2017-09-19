---
title: Mac下 Latex 环境配置（含中文）
date: 2016-11-02 22:03:08
tags: 
- Latex
categories:
- Tools
---

Latex作为强力的排版和科学论文写作工具，在日常学习工作中作用强大，表达能力远超 Markdown，Org-mode 等简洁写作格式。本文记录了Mac下Latex环境的配置。

<!-- more -->

Mac下配置Latex环境有很多种方式：

1. 使用诸如 MacTex 等集成开发环境，安装大小2G+；
2. 使用文本编辑器加编译器 MacTex Basic 包的方式，安装大小100M左右。

而文本编辑器的选择有很多种，常见的有 Sublime Text，Atom，Emacs或者Vi。

## Sublime Text + LaTeX Tools

安装步骤如下：

1. 安装 Sublime Text 的 Package Control
2. 安装 MacTex Basic包
3. Sublime Text中 `Ctrl-Shift-P` 调出 Package Control，输入 Install，选择 LaTexTools 安装
4. 安装 Skim，它可以在 tex 编译后实时预览
5. 在 Skim 选项-同步，PDF-Tex同步中设置 Sublime Text

此时 Latex 的环境基本正常，使用 `COMMAND+B` 即可构建。现在讲述如何搭建中文环境：

打开终端，运行：

	sudo tlmgr update --self
	sudo tlmgr install latexmk

在ST里打开LaTeXTools.sublime-settings（也就是LaTeXTools的用户设置，如果你是从旧版本升级上来或者担心这个配置文件出现问题，可以依次点击Preferences——Package Settings——LaTeXTools——Reconfigure LaTeXTools and migrate settings重建配置文件），在builder-settings下面新增两项配置：

	"program" : "xelatex",
	"command" : ["latexmk", "-cd", "-e", "$pdflatex = 'xelatex -interaction=nonstopmode -synctex=1 %S %O'", "-f", "-pdf"],

另外注意之前应该有"builder": "default"（或直接设置为空或”traditional”）。

## FAQ

1. 如何安装拓展包如 algorithmicx 等？

	sudo tlmgr install xxxx

2. 遇到以下问题 `Unknown directive ...containerchecksum c59200574a316416a23695c258edf3a32531fbda43ccdc09360ee105c3f07f9fb77df17c4ba4c2ea4f3a5ea6667e064b51e3d8c2fe6c984ba3e71b4e32716955... , please fix it! at /usr/local/texlive/2015basic/tlpkg/TeXLive/TLPOBJ.pm line 210, <$retfh> line 5579.`

	tlmgr option repository ftp://tug.org/historic/systems/texlive/2015/tlnet-final

[Reference](http://tex.stackexchange.com/questions/313768/why-getting-this-error-tlmgr-unknown-directive)

3. 中文环境

[Reference](http://blog.jqian.net/post/xelatex.html)

4. algorithm

	sudo tlmgr install relsize
	sudo tlmgr install algorithm2e