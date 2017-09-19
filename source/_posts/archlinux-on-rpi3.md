---
title: 树莓派3安装ArchLinux
date: 2017-09-18 22:05:27
tags: ArchLinux
categories: 树莓派 
---

作为热门的开源硬件，树莓派的爱好者们乐于拿它做各种新奇的应用，甚至可以拿32个树莓派搭个计算能力强悍的集群。树莓派3开始提供板载Wi-Fi和蓝牙功能，在性能上提升不少。而树莓派的操作系统层出不穷，包括常用的Raspbian，Ubuntu Mate等，本文记录了简洁优雅的ArchLinux-ARM在树莓派上的安装。安装过程基于Linux系统。

<!-- more -->

### 下载镜像

可以从国内的各大开源镜像站下载，比如清华、网易的开源镜像站。

[Tsinghua Tuna](https://mirrors.tuna.tsinghua.edu.cn/archlinuxarm/os/)站点下有各种可选的(rpi)镜像。本文使用的是`ArchLinuxARM-rpi-3-latest.tar.gz`。可以使用如下方式下载:

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/archlinuxarm/os/ArchLinuxARM-rpi-3-latest.tar.gz
```

### 分区

准备好镜像后，接下需要将SD卡分区。按照官方wiki说明，分区如下：

1. 一个100M大小的vfat分区，挂载boot
2. 剩余空间是ext4分区，挂载/

将SD插入后，找到对应的设备文件`/dev/sdX`后，使用`fdisk /dev/sdX`:

1. `o` 来清除原来分区表
2. `n` 来新建分区，类型选择`p`，输入`1`作为设备第一个分区，输入`ENTER`选择默认开始扇区，`+100M`表示分区大小100M
3. 输入`t`，然后输入`c`设置分区类型为`W95 FAT32 (LBA)`
4. `n` 来新建分区，类型选择`p`，以后都使用`ENTER`使用默认值
5. `w` 保存分区表

接下来新建目录挂载FAT、EXT4分区：

```bash
mkfs.vfat /dev/sdX1
mkfs.ext4 /dev/sdX2
mkdir root
mkdir boot
mount /dev/sdX1 boot
mount /dev/sdX2 root
```

注意，vfat需要dos文件系统支持，如 ArchLinux 需要安装`dosfstools`。

### 部署镜像

将下载的镜像解压，然后将解压后的boot目录拷贝到boot分区。

```bash
bsdtar -xpf ArchLinuxARM-rpi-latest.tar.gz -C root
sync
mv root/boot/* boot
umount boot root
```

接下来将SD卡插入树莓派接入电源就可以启动了。接下来讲如何连接Wi-Fi，为了安装软件包，需要以太网连接。

默认的用户是：

- 账号 root 密码 root ，该账户无法远程登录
- 账号 alarm 密码 alarm

可以使用`Ctrl+Alt+ F1-F7`来切换终端，其中`F7`表示GUI，该系统没有X图形界面，有需要的童鞋请自行安装。

### 修改镜像源

将镜像源修改为国内的，可以加快下载速度，尤其是教育网用户使用教育网镜像站。

```bash
vi /etc/pacman.d/mirrolist
```

以清华镜像站为例，添加如下记录：

```
Server = https://mirrors.tuna.tsinghua.edu.cn/archlinuxarm/$arch/$repo
```

注：该文件中越靠前的记录优先级越高。

使用`pacman -Syy`更新本地数据库。

### 连接Wi-Fi

现在ArchLinuxARM可以使用了，接下来可以利用板载Wi-Fi。这一部分官方wiki上没有给出，参考ArchLinux的netctl命令，整个过程如下：

检查驱动

```bash
ifconfig -a
```

可以看到 wlan0 接口。接下来安装一些必要软件包：

```bash
pacman -S wpa_supplicant
```

对于一个无线网络，ESSID是网络名称，KEY是Wi-Fi密码。使用`wpa_passphrase <ESSID> <KEY>`生成256bit PSK，它由KEY和SSID经由标准算法计算而成，这一步是为了避免密码明文存储，生成结果如下：

```json
network={
  ssid="your_essid"
  #psk="passphrase"
  psk=64cf3ced850ecef39197bb7b7b301fc39437a6aa6c6a599d0534b16af578e04a
}
```

接下来新建netctl配置，如 tplink，修改`vi /etc/netctl/tplink`，内容如下：

```
Description='<YOUR DESCRIPTION>'
Interface=wlan0
Connection=wireless
Security=wpa
IP=dhcp
ESSID=<ESSID>
Key=\"<PSK>
```

其中`<PSK>`就是上面生成的psk。

开启DHCP服务`dhcpcd`。

最后使用 `netctl enable tplink` 作为系统服务。使用 `netctl start tplink` 连接Wi-Fi。可以拔掉网线了。

### 参考资料

- [ArchLinux RPI3](https://archlinuxarm.org/platforms/armv8/broadcom/raspberry-pi-3)
- [Netctl](https://wiki.archlinux.org/index.php/Netctl)