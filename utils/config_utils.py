import os.path as osp
import tempfile
import re
import shutil
import sys
import ast
from importlib import import_module

'''
Thanks the code from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py wrote by Open-MMLab.
The `Config` class here uses some parts of this reference.
'''


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """Check if the input file exists."""
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class Config:
    """
    配置类
    """
    def __init__(self, cfg_dict=None, filename=None):
        """
        构造函数，初始化Config对象
        :param cfg_dict: 配置字典，默认为空字典
        :param filename: 配置文件路径
        """
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')

        if filename is not None:  # 读取文件内容并更新cfg_dict
            cfg_dict = self._file2dict(filename)
            filename = filename

        # 使用super调用父类的__setattr__方法，设置_config_dict和_filename属性
        super(Config, self).__setattr__('_cfg_dict', cfg_dict)
        super(Config, self).__setattr__('_filename', filename)

    @staticmethod
    def _validate_py_syntax(filename):
        """
        验证Python文件的语法是否正确
        :param filename: 配置文件路径
        :raises SyntaxError: 如果文件语法有误，抛出SyntaxError
        """
        with open(filename, 'r') as f:  # 打开并读取文件内容
            content = f.read()
        try:
            ast.parse(content)  # 使用ast.parse解析文件内容，检查语法是否正确
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _file2dict(filename):
        """
        读取配置文件并转换为字典
        :param filename: 配置文件路径
        :param use_predefined_variables: 是否替换预定义变量
        :return: 配置字典
        """
        filename = osp.abspath(osp.expanduser(filename))  # 获取文件的绝对路径
        check_file_exist(filename)  # 检查文件是否存在
        fileExtname = osp.splitext(filename)[1]  # 获取文件扩展名，确保是.py文件
        if fileExtname not in ['.py']:
            raise IOError('Only py type are supported now!')
        content = None
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证文件内容的Python语法是否正确
        Config._validate_py_syntax(filename)
        # 执行文件内容，将内容作为Python代码执行
        exec_globals = {}
        exec(content, exec_globals)
        # 从执行结果中提取出非内置变量，形成字典
        cfg_dict = {
            name: value
            for name, value in exec_globals.items()
            if not name.startswith('__')
        }
        return cfg_dict

    @staticmethod
    def fromfile(filename):
        """
        从文件加载配置并返回Config对象
        :param filename: 配置文件路径
        :return: 配置对象Config
        """
        cfg_dict = Config._file2dict(filename)  # 读取文件并转换为字典
        return Config(cfg_dict, filename=filename)  # 返回Config对象
