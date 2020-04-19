#ifndef LIST_H
#define LIST_H

#define LIST_END -1

#define OBJ void*
typedef struct {
	union {
		int val;//整形值
		OBJ obj;//对象地址
	} ele;//元素
	struct ListNode* next;//下一个node
}ListNode;
typedef struct {
	int eleLen;//链表长度
	int eleSize;//元素大小
	struct ListNode* first;//起始元素
	struct ListNode* last;//末尾元素
} List;//链表头



List* ListCreat(int eleSize);
//在链表第n个元素后插入一个元素,末尾为LIST_END
void ListAdd(List* list, OBJ obj, int n);
//获取第n个元素
OBJ ListGet(List* list, int n);
//将所有元素以数组形式返回并销毁链表
void* ListGetAll(List* list);
//销毁所有链表元素
void ListDeleAll(List* list);

#endif
