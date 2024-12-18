开发中，不稳定

main.py, index_human.html, index.html是为了机器人对战网页观察效果的。

python main.py


对无界面对战来说，tcg.py是游戏启动入口
python tcg.py

ai.py是实现各种player的核心代码
正在实现LMPlayer，是基于Douzero的DMC方法，只需要训练一个Q网络即可。这里的特色是加上了一个语言模型+一个微调的字符串到数字的头来得到一个良好的参数初始化。还没完成。
现在score_layer_weights.pth中保存了Qwen/Qwen2.5-Coder-1.5B-Instruct对应的转换头。训练方式是
LMTrainer 中
self.run_mode = ["train_loss_head", "train_rl"][0]
然后跑
python tcg.py

model.py是跟机器学习相关的代码
log.py提供了日志记录接口
aiapisync.py是用apikey调用LLM的代码

dice_pay.py是用来列举和剪枝所有dice_pay方案和roll_dice方案的，由波比提供，但是模块本身没有完善，也还没接上和使用。
