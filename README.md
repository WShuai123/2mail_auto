# 2mail_auto
After the model training is completed, the results will be automatically sent to the mailbox

因为我经常需要在云服务器上跑数据，训练模型，训练时间很长，经常需要不定时打开看看是否训练完成。
因此，这些代码可以实现自动发送邮件的功能。网络模型训练完自动把训练结果发送到指定的邮箱。

但是仓库里的只是最基础的代码，若要实现以上两种功能，需要在你训练的模型的训练脚本里进行调整。
如demo.py所示。

demo.py实现的是全部训练完成后一次发送每个epoch的训练结果。
如要每训练完一个epoch就发送一次训练结果，自己稍微改一下就行。

注：demo.py无法运行，没有上传调用的py文件和数据集。demo.py只是swin transformer的训练脚本。

demo.py添加的自动发邮件的代码有：
+ 第13-38行
+ 第111行
+ 第117，123行
+ 第134行

最终实现的效果如图：
https://github.com/WShuai123/2mail_auto/blob/main/test.jpg?raw=true
