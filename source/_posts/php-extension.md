---
title: php-extension
date: 2019-05-05 19:48:59
tags: php
categories:
- php
- c/c++
- zend
---

# 钩子函数

MINIT
MINIT_FUNCTION

# 全局资源

ZEND_BEGIN_MODULE_GLOBALS

ZEND_END_MODULE_GLOBALS

ZEND_DECLARE_MODULE_GLOBALS

ZEND_MODULE_GLOBALS_ACCESSOR()

一个拓展中可以定义多个全局变量结构体

# ini配置
PHPRC

PHP_INI_BEGIN()

PHP_INI_END();

STD_PHP_INI_ENTRY(name, default_value, modifiable, on_modify, property_name, struct_type, struct_ptr)

- name 配置项
- default_value 必须是字符串表示的值
- modifiable  ZEND_INI_USER/SYSTEM/PERDIR/ALL PHP脚本中可以修改、php.ini/httpd.conf可以、除此之外，还有.htaccess
- on_modify OnUpdateBool/Long/GEZeroi/Real/String/StringUnempty 可以自定义
- property_name 结构体成员
- struct_type 结构体类型
- struct_ptr 结构体地址

REGISTER_INI_ENTRIES();

ZEND_INI_MH(name)

mh_arg2 base
mh_arg1 offset
```c
#ifndef ZTS
char* base = (char*) mh_arg2;
#else
char* base;
base = (char*) ts_resource(*((int*)mh_arg2));
#endif
```

# 函数

## 内部函数注册

不管是内部函数还是用户自定义函数，最终都会注册到 EG(function_table) 中。

ZEND_FUNCTION/ PHP_FUNCTION  zif_xxx()

(zend_execute_data* execute_data, zval* return_val)

zend_function_entry => zend_module_entry->functions

ZEND/PHP_FE()

PHP_FE_END

## 参数解析

用户自定义函数编译时会为每个参数创建一个 zend_arg_info 结构，用来记录 参数名称、是否引用传参、是否可变参数等信息

存储上函数参数和局部变量没有区别，分配在 zend_execute_data 上

调用函数时，首先进行参数传递，按照参数次序将value从函数调用空间传递到 zend_execute_data，之后像访问局部变量一样访问参数

内部函数最大区别，局部变量是c变量，不是分配在 zend_execute_data 上

Zend/Zend_API.h

ZEND_API int zend_parse_parameters(int num_args, const char* type_spec, ...);

- num_args 函数调用时实际传递的参数，ZEND_NUM_ARGS() 获得， zend_execute_data->This.u2.num_args
- type_spec 解析规则
- 解析到的变量，与上面的配合使用

解析时除了bool、整型、浮点型是直接拷贝，其余解析到的变量只能是指针，因为解析过程取得是ED上的参数 zval 或者具体 value 的地址

PHP5.x


PHP7.x

ZEND_PARSE_PARAMETERS_START(2,3)
ZEND_PARSE_PARAMETERS_END();

- l/L zend_long L将超出范围解析为long最大值，而l将报错

如果标识符后面加 ! 表示运行该参数传NULL，则必须再提供一个 zend_bool 变量的地址。如果为NULL，解析到的为0，bool设置为1.如果不加！，传NULL则设置为0，无法判断是否传0还是NULL

Z_PARAM_LONG(_EX)(dst[, is_null, check_null, separate]) l
Z_PARAM_STRICT_LONG(_EX)(dst[, xx,xx,xx])  L

- b 支持！

Z_PARAM_BOOL(dst) ZPARAM_BOOL_EX(dst, is_null, check_null, separate)

- separate 指定参数的value是否分离，定义为1，则参数为string、array时将复制 value

- d !

Z_PARAM_DOUBLE[_EX]()

- s S p P 后两者主要解析路径 s Char* size_t S zend_string* 
Z_PARAM_STRING(dst, dst_len) s
Z_PARAM_STRING(dst, dst_len, check_null, separate)
Z_PARAM_STR  S
Z_PARAM_PATH()
Z_PARAM_PATH_STR()

- a A h H zval地址 / HashTable 地址
A/H 支持对象,，小写报错
A按照对象解析，H调用对象 get_propreties() 获得属性数组

Z_PARAM_ARRAY(_EX)
Z_PARAM_ARRAY_OR_OBJECT(_EX) 两者参数一样。前者其实参数并没任何作用
Z_PARAM_ARRAY_HT_(EX)

- o O
解析到的变量只能是 zval 的地址，无法解析到 zend_object()
O 指定类或者子类对象

Z_PARAM_OBJECT(_EX)
Z_PARAM_OBJECT_OF_CLASS(dst, _ce, check_null, separate)

- r 只能解析到 zval 无法到 zend_resource
!与string相同

Z_PARAM_RESOURCE(dst) / Z_PARAM_RESOURCE_EX(dst, check_null, separate)

- C 参数是一个类则可以解析到 zend_class_entry; ce指定类，则只有存在父子关系才能解析，如果只是根据参数获取类型记得将ce初始化为NULL

Z_PARAM_CLASS(_EX)

- f 回调类型

函数或者成员方法，函数名字符串、array（对象/类，成员方法） 则可以通过f解析出 zend_fcall_info 注意不是 zend_fcall_info *。

zend_fcall_info
zend_fcall_info_cache 这俩不能是指针


my_func("func_name")
my_func(array('class_name', 'static_method'))
my_func(array($obj, 'method'))

Z_PARAM_FUNC(dst_fci, dst_fcc) Z_PARAM_FUNC_EX(dst_fci, dst_fcc, check_null, separate)

- z 任意类型
Z_PARAM_ZVAL(dst) Z_PARAM_ZVAL(dst, check_null, separate)

- 其他
| 表示可选参数
新的解析方式 Z_PARAM_OPTIONAL

+ * 前者一个，后者0+，解析到 ZVAL* 数组，通过一个整型参数获取具体数量。

Z_PARAM_VARIADIC(spec, dst, dst_num) spec = "*"/"+"

long/bool/double/string/array、hashtable/object/class/fcall/z/resource

## 引用传参

ZEND_BEGIN_ARG_INFO
ZEND_BEGIN_ARG_INFO_EX(name,_unused[, return_reference, required_num_args])
ZEND_END_ARG_INFO

ZEND_ARG_INFO(pass_by_ref, name)
ZEND_ARG_PASS_INFO(pass_by_ref)
ZEND_ARG_OBJ_INFO(pass_by_ref, name, classname, allow_null)
ZEND_ARG_ARRAY_INFO(pass_by_ref, name, allow_null)
ZEND_ARG_CALLABLE_INFO(pass_by_ref, name, allow_null)
ZEND_ARG_TYPE_NFO(pass_by_ref, name, type_hint, allow_null)
ZEND_ARG_VARIADIC_INFO(pass_by_ref, name)

## 函数返回值

b IS_FALSE/IS_TRUE

RETURN_BOOL(b)
RETURN_NULL()
RETURN_LONG(l)
RETURN_DOUBLE(d)
RETURN_STR(s) zend_string*
RETRUN_INTERNED_STR(s)   zend_string* 不会被回收
RETURN_NEW_STR(s) zend_string*
RETRUN_STR_COPY(S) zend_string*
RETRUN_STRING(s) 
RETURN_STRING(s, l)
RETURN_EMPTY_STRING()
RETRUN_RES(r)
RETURN_ARR(a)
RETURN_OBJ(o)
RETRUN_ZVAL(zv, copy, dtor)
RETURN_FALSE
RETURN_TRUE

## 函数调用
ZEND_API int call_user_function(HashTable* function_table, zval* object
, zval* function_name, zval* retval_ptr, uint32_t param_count, zval params[])

zend_string_init(s, strlen(), 0)

zend_string_release(s)

函数传参时不会硬拷贝value，而是增加参数value的引用计数，然后在函数return阶段把引用再减掉。call_user_function 替我们完成这个工作

zend_call_function(zend_fcall_info*, NULL)

# ZVal操作

## 创建及获取

ZVAL_XXX() 设置不同类型的zval
Z_XXX(zval)/ Z_XXX_P 获取不同类型zval的value

Z_TYPE(zval) Z_TYPE_P(zval_p)获取类型，实际上是 zval.u1.v.type

设置时不能只修改这个type，而是要设置typeinfo，因为涉及到 是否使用引用计数、是否可被垃圾回收、是否可被复制

#### undef null
ZVAL_UNDEF
ZVAL_NULL

Z_ISNUDEF
Z_ISUNDEF_P
Z_ISNULL
Z_ISNULL_P

#### bool
ZVAL_BOOL(z, b) b IS_TURE/FALSE
ZVAL_FALSE
ZVAL_TRUE

Z_TYPE_P获取布尔值直接判断

#### long double
ZVAL_LONG(z, v)  zend_long double
ZVAL_DOUBLE(z, v)

Z_LVAL(_P)
Z_DVAL(_P)

#### string
ZVAL_STR(z, s) s zend_string  不会将s复制一份，只是将s地址设置到新的zval，s的引用计数不会增加。

ZVAL_NEW_STR(z, s) 不支持内部字符串，如果传入内部，则生产普通字符串


ZVAL_STR_COPY(z, s) 与 ZVAL_STR 操作一致，不过会根据类型增加引用计数

Z_STR
Z_STR_P zend_string
Z_STRVAL(_P) char *  zend_string->val
Z_STRLEN_P zend_string->le
Z_STRHASH_P zens_string->h

#### array
ZVAL_ARR(z, a) 不会增加a的引用计数
ZVAL_NEW_ARR(z) 分配了array内存，但是没有初始化，还不能使用

Z_ARR_P
Z_ARRVAL_P

#### object
ZVAL_OBJ
Z_OBJ_P

Z_OBJ_HT_P  handlers
Z_OBJ_HANDLER_P  read_property/write_property read only
Z_OBJ_HANDLE_P handle
Z_OBJCE_P ce
Z_OBJPROP_P  get_property

#### resource
ZVAL_RES
Z_RES
Z_RES_P

Z_RES_HANDLE_P 获取资源的handle

#### reference
ZVAL_REF(z, r)
ZVAL_NEW_REF(z, r)
Z_REF
Z_REF_P
Z_REFVAL_P

#### others
Z_INDIRECT_P(zval)   (zval).value.zv
Z_CE_P  zend_class_entry
Z_FUNC_P zend_function_entry
Z_PTR_P zval保存的ptr （zval).value.ptr

## 变量复制
ZVAL_COPY(z,v) z目标，复制后增加vlaue的引用计数
ZVAL_COPY_VALUE(z, v) 不增加计数

## 引用计数
Z_TRY_ADDREF_P(arr)

什么时候需要考虑？
操作的是与PHP用户空间相关的变量，包括对用户空间变量的修改、赋值。需要明确的一点是，引用计数是解决指向同一个value的问题，所以在 PHP 中来回传递zval的时候就需要考虑是否修改引用计数

变量赋值
数组操作 针对元素
函数调用 虽然变化,但是由内核完成,拓展不需要处理
成员属性 当把一个变量赋值给对象的成员属性时,需要增加引用计数

Z_REFCOUNT_P
Z_SET_REFCOUNT_P
Z_ADDREF_P
Z_DELREF_P
Z_REFCOUNT
Z_SET_REFCOUNT
Z_ADDREF
Z_DELREF

Z_TRY_ADDREF_P
Z_TRY_DELREF_P
Z_TRY_ADDREF
Z_TRY_DELREF

//zend_value
GC_REFCOUNT
Z_REFCOUNTED 是否用到了引用计数
Z_REFCOUNTED_P
Z_COUNTED 获取zend_refcounted 头部
Z_COUNTED_P

#### string

zend_string_init(const char* str, size_t len, int persistent)
zend_string_copy(zend_string *s)
zend_string_dup(zend_string *s, int persistent)
zend_string_realloc(zend_string* s, size_t len, int persistent)
zend_string_extend() 同上，但是len不能小于s的长度
zend_string_truncate() 不能大于s长度
zend_string_refcount
zend_string_addref
zend_string_delref
zend_string_release
zend_string_free
zend_string_equals

zend_string_equals_ci

ZSTR_VAL
ZSTR_LEN
ZSTR_H  前三者获取字段
ZSTR_HASH zend_string_hash_val(zstr)