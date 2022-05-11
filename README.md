# 2mail_auto
After the model training is completed, the results will be automatically sent to the mailbox
实现自动发送邮件的功能。

因为我经常需要在云服务器上跑数据，训练模型，训练时间很长，经常需要不定时打开看看是否训练完成。
因此，这些代码可以实现：
+ 训练完一个epoch发送一个训练结果
+ 训练完全部的epoch，一次发送全部的训练结果

但是仓库里的只是最基础的代码，若要实现以上两种功能，需要在你训练的模型的训练脚本里进行调整。
如demo.py所示
