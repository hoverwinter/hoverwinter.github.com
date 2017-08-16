---
title: MAC下配置Scala开发环境
date: 2017-07-29
tags: Scala 
categories:
- Tools
---

决定入Scala坑，所以记录一下环境的搭建过程。

<!-- more -->

## Homebrew 安装 Scala

直接使用

```shell
brew install scala
brew install sbt
```

然后命令行执行 sbt ，下载相关资源。

即可

## IDEA配置Scala环境

从IDEA设置中安装 Plugin：

- sbt
- scala

安装完成后重启。

然后在设置里面搜索 sbt，将 Launcher设置为 Custom `/usr/local/Cellar/sbt/xxxx/libexec/sbt-launcher.jar`，这个目录可以通过 `which scala`和 `ll xxx`来找到。

设置项目的 build.properties 的版本为自己下载的sbt版本。将 build.sbt 中 scalaVersion 设置为自己下载的版本。并添加如下一行：

```
scalaHome := Some(file("/usr/local/Cellar/scala/2.11.8/idea"))
```

最后在右侧的sbt中选择刷新即可。

新建Scala文件会提示没有SDK，在Project Structure中 Global libraries中添加 scala-SDK 即可。 

这个时候选择版本下载也是可以的。也可以自己下载然后手动添加。brew install由于mac 目录的原因，需要link之类的操作来使用。