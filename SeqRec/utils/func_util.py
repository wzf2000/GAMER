import inspect
from loguru import logger
from typing import TypeVar
from functools import wraps

T = TypeVar("T")


def log_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取函数签名
        sig = inspect.signature(func)
        # 绑定参数到签名
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()  # 应用默认值
        # 转换为参数字典
        param_dict = dict(bound_args.arguments)
        logger.info(f"Calling {func.__name__} with arguments: {param_dict}")
        if 'self' in param_dict:
            class_instance = param_dict.pop('self')  # 移除self参数
            class_instance.param_dict = param_dict
        return func(*args, **kwargs)
    return wrapper


def create_meta_class(
    base_class_name: str,
    methods_to_decorate: tuple[str],
    decorator: callable
):
    class CustomMeta(type):
        def __new__(cls, name, bases, attrs):
            if name != base_class_name:
                for method_name in methods_to_decorate:
                    if method_name in attrs and callable(attrs[method_name]):
                        attrs[method_name] = decorator(attrs[method_name])
            return super().__new__(cls, name, bases, attrs)

    return CustomMeta


def subclasses_recursive(cls: type[T]) -> list[type[T]]:
    """
    Recursively find all subclasses of a given class.
    """
    subclasses: list[type[T]] = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(subclasses_recursive(subclass))
    return subclasses
