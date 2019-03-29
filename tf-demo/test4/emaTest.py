"""
滑动平均 exponential moving average
记录每个参数一段时间过往值的平均
当参数w或b发生变化时, ema缓慢追随(类似于cpu load average), 因此也被称为影子值.

ema_new = 衰减率 * ema + (1 - 衰减率) * 参数
其中, 衰减率 = min(moving_average_decay, (1+轮数)/(10+轮数))

初始情况下, moving_average_decay会给一个较大的值, 例如0.99, 参数, 轮数, 滑动平均值都为0.
第1轮训练后, 假设w更新成了1,
ema = min(0.99, 1/10) * 0 + (1 - min(0.99, 1/10)) * 1 = (1 - 0.1) * 1 = 0.9
第100轮训练后, 假设w更新成了10,
ema = min(0.99, 101/110) * 0.9 + (1 - min(0.99, 101/110)) * 10 = 0.826 + 0.818 = 1.644
...以此类推

"""

import tensorflow as tf


def ema_test():
    w = tf.Variable(0, dtype=tf.float32)
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False)

    moving_average_decay = 0.99
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # ema.apply()参数为一个列表, 每次运行sess.run(ema_op)时, 对apply参数列表中的元素求滑动平均值
    # 使用tf.trainable_variables()自动获取所有需要训练的参数
    ema_op = ema.apply(tf.trainable_variables())
    """
    # 实际工程中, 常把计算滑动平均和训练过程绑定在一个训练节点
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # ema.average(参数名称) 查看某个参数的滑动平均值
        # 打印参数和参数的滑动平均值
        print("初始")
        print(sess.run([w, ema.average(w)]))
        # 给参数w赋值
        sess.run(tf.assign(w, 1))
        print("w设置初始值")
        # 计算滑动平均
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))

        # 假设经过了100轮迭代, 此时w变成了10
        print("经过100轮迭代")
        sess.run(tf.assign(w, 10))
        sess.run(tf.assign(global_step, 100))
        # 再次计算滑动平均
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))

        # 每次计算, 都会更新(根据之前的平均), 因为这里的参数值并没有改变, 所以可以看到滑动平均值在向参数值逼近
        print("更新")
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))
        sess.run(ema_op)
        print(sess.run([w, ema.average(w)]))

    return None


if __name__ == '__main__':
    ema_test()
