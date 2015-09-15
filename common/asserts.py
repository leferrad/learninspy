__author__ = 'leferrad'

# Decoradores para validacion:
def assert_typearray(func):
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if not isinstance(args[1], type(args[0])):
            raise Exception('Solo se puede operar con arreglos del mismo tipo!')
        else:
            return func(*args)
    return func_assert


def assert_samedimension(func):
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if args[0].shape() != args[1].shape():
            raise Exception('Arreglos con dimensiones distintas!')
        else:
            return func(*args)
    return func_assert


def assert_matchdimension(func):
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if args[0].cols != args[1].rows:
            raise Exception('Arreglos con dimensiones que no coinciden!')
        else:
            return func(*args)
    return func_assert
