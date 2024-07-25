# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def print99():
    for i in range(1,10):
        for k in range(1,i+1):
            print(f"{k} x {i} = {i * k}\t", end="")
            #print('{}x{}={}\\t'.format(i, j, i * j), end='')
        print()

def reverse():
    my_list = [1,2,6,3,7]
    # reverse=True : 表示降序，默认不写的话表示升序
    my_list.sort(reverse=True)
    print(my_list)
    
def sort():
    my_list = [{'name':'zs','age':20},{'name':'ls','age':19}]
    my_list.sort(key=lambda item: item['age'], reverse=True)
    print(my_list)

    def get_value(item):
        return item['age']

    my_list.sort(key=get_value, reverse=True)
    print(my_list)
    
    
def cp():
    old_file = open('./source.txt', 'rb')
    new_file = open('./target.txt', 'wb')

    # 文件操作
    while True:
    #     1024 : 读取1024字节的数据
        file_data = old_file.read(1024)
    #     判断数据是否读取完成
        if len(file_data) == 0:
            break
        new_file.write(file_data)

    # 关闭文件
    old_file.close()
    new_file.close()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sort()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
