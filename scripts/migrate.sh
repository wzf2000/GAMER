#!/bin/bash

# 将第一个参数对应的文件建立软链接到第二个参数对应的文件
# 如果第一个参数是软链接，则跳过，不进行任何操作
# 如果第二个参数存在，则先对比两个文件的内容，如果内容不同，则提示不同并退出
# 如果内容相同，则删除第二个参数对应的文件，并建立软链接到第一个参数对应的文件
# 如果第二个参数不存在，则不进行任何操作
function soft_link_file() {
    if [ -L $1 ]; then
        # 设置蓝色输出，前置link symbol emoji 🔗
        echo -e "\033[34m🔗 $1 是软链接，跳过\033[0m"
        return 0
    fi
    if [ -f $2 ]; then
        if [ "$(cat $1)" != "$(cat $2)" ]; then
            # 设置红色输出，前置diff symbol emoji ⚖️
            echo -e "\033[31m⚖️  $1 和 $2 文件内容不同\033[0m"
            return 1
        else
            # 设置绿色输出，前置success symbol emoji ✅
            echo -e "\033[32m✅ $1 和 $2 文件内容相同，删除 $2 并建立软链接到 $1\033[0m"
            rm -f $2
            # 获得 $1 相对于 $2 所在目录的相对路径
            relative_path=$(realpath --relative-to=$(dirname $2) $1)
            ln -s $relative_path $2
            return 0
        fi
    else
        # 设置蓝色输出，前置skip symbol emoji ⏩
        echo -e "\033[34m⏩ $2 不存在，不进行任何操作\033[0m"
        return 0
    fi
}

# 判断第一个参数对应的是否是目录，如果是文件，则调用soft_link_file函数，否则遍历其下所有文件，并调用soft_link_dir或soft_link_file函数
function soft_link_dir() {
    for file in $(ls $1); do
        if [ -f "$1/$file" ]; then
            soft_link_file "$1/$file" "$2/$file"
        else
            soft_link_dir "$1/$file" "$2/$file"
        fi
    done
}

if [ -d $1 ]; then
    soft_link_dir $1 $2
else
    soft_link_file $1 $2
fi
