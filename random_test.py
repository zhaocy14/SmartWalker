import threading
import time
#
#
# class A(object):
#     def __init__(self):
#         self.threading_flag = True
#
# a = A()
#
# def loop_1(A):
#     while True:
#         if A.threading_flag:
#             print("hello")
#             pass
#         else:
#             print("wait for the flag")
#             time.sleep(1)
#
# t = threading.Thread(target=loop_1, args=(a,))
# t.start()
# time.sleep(1)
# a.threading_flag = False
# print("thread_flag is set to False")
# time.sleep(5)
# a.threading_flag = True
# print("thread_flag is set to False")


e1 = threading.Event()
e2 = threading.Event()

def loop(e:threading.Event,s:str):
    while True:
        e.wait()
        print("running %s"%s)


t1 = threading.Thread(target=loop,args=(e1,"1"))
t2 = threading.Thread(target=loop,args=(e1,"2"))
t1.start()
t2.start()

print("main start!")
time.sleep(1)
e1.set()
e1.set()
e1.set()
e1.set()
time.sleep(2)
e1.clear()
time.sleep(1)
e2.set()
time.sleep(2)
e2.clear()
print(e1.is_set(),e2.is_set())