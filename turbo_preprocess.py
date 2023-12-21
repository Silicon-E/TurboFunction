import re
import os
import sys
import json
import argparse
from typing import Callable


"""
Turbo: A .mcfunction preprocessor
Turbo speeds up your .mcfunction coding with constructs including constants, macros, code blocks, and local functions.

by Silas Barber, 2023.

python turbo_preprocess.py
    Recursively searches for all .mcfunction files in the data/ directory and processes them.

python turbo_preprocess.py <input_path>
    Processes the .mcfunction file located at <input_path>.
"""


#   TODO
# Rebuild all when this python script changes.
# Rebuild dependee functions when a dependency function changes.
# Add -clean option.
# Clean output files for a compilation unit before outputting them. This prevents obsolete output files from persisting if they aren't overwritten.
# Maybe automatically clean output files for sources that no longer exist.

syntax = {
    'src_extension': ' SRC.mcfunction',
    'dest_extension': '.mcfunction',
    'directive_prefix': '##',
    'embed_open': '%(',
    'embed_close': ')',
}

cache_version = 1

def re_union(*patterns : re.Pattern):
    return '|'.join(compiled_pattern.pattern for compiled_pattern in patterns)

def normalize_path(path):
    return path.replace('\\', '/')

def path_to_id(path):
    path = normalize_path(path)
    relpath = normalize_path(os.path.relpath(path, 'data'))
    namespace = relpath.split('/', 1)[0]
    name = os.path.splitext(relpath)[0].removeprefix(f'{namespace}/functions/')
    return (namespace, name)

class CompilationUnit:
    def __init__(self, source_path : str, path : str, parent : 'CompilationUnit' = None) -> None:
        self.source_path = source_path
        self.path = path
        self.dependents : set[CompilationUnit] = set()
        self.parent = parent
        (self.namespace, self.id) = path_to_id(path)
        self.anonymous_child_count = 0
        'Used to give numbered names to anonymous child functions.'
        pass

    def make_child(self, name : str = None):
        if name == None:
            name = str(self.anonymous_child_count)
            self.anonymous_child_count += 1
        # Create a unique path for this new function.
        path = self.path.removesuffix(syntax['dest_extension'])
        # If the parent unit has no parent (it is top-level), put the new child unit in a __turbo folder alongside the parent unit.
        # Otherwise, put it in the same directory as the parent.
        if not self.parent:
            head, tail = os.path.split(path)
            path = f'{head}/__turbo/{tail}'
        path += f'_{name}{syntax["dest_extension"]}'
        return CompilationUnit(None, path, parent=self)

class ParserError(Exception):
    def __init__(self, message : str, pos : tuple[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.pos = pos
    
    def __str__(self) -> str:
        return f'{self.pos}: {self.message}'

class Symbol:
    def __init__(self, name : str) -> None:
        self.name = name
    
    def invoke(self, args : list) -> str:
        raise ParserError(f"Cannot invoke a {type(self).__name__} with parameters, but got parameters: {repr(args)}")

    def insert(self) -> str:
        raise ParserError(f"Cannot insert a {type(self).__name__} without parameters, but no parameters were given.")

class SymbolArg(Symbol):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.value = None

class SymbolMacro(Symbol): # TODO: Delete.
    def __init__(self, name: str, argnames: list[str], parent_scope: 'Scope') -> None:
        super().__init__(name)
        self.argnames = argnames
        self.scope = Scope(parent_scope)
        for argname in argnames:
            self.scope.add_symbol(SymbolArg(argname))
    
    def output(self, lines : list[str], args : list[str]):
        if len(args) != len(self.argnames):
            raise ParserError(f"Macro {self.name} expected {len(self.argnames)} arguments, but got {len(args)}.")
        for i, argname in enumerate(self.argnames):
            self.scope.symbols[argname].value = args[i]
        for command in self.scope.code:
            command.output(self.scope, lines)

class SymbolVariable(Symbol):
    def __init__(self, value : str) -> None:
        self.value = value

    def evaluate(self) -> str:
        return self.value

class SymbolTemplate(Symbol):
    def __init__(self, parent_scope : 'Scope', argnames : list[str], commands : list['Command']) -> None:
        self.parent_scope
        self.argnames = argnames
        self.commands = commands

    def insert(self):
        return self

    def invoke(self, args: list):
        if len(args) != len(self.argnames):
            raise ParserError(f"Template {self.name} expected {len(self.argnames)} arguments, but got {len(args)}: {repr(args)}")
        
        inner_scope = Scope(self.parent_scope)
        for i, argname in enumerate(self.argnames):
            arg = args[i]
            if isinstance(arg, str):
                arg = SymbolVariable(arg)
            inner_scope.add_symbol(argname, arg)

        inner_lines = []
        for command in self.commands:
            command.output(inner_scope, inner_lines)
        return '\n'.join(inner_lines)

class SymbolFunction(Symbol):
    def __init__(self, name: str, parent_scope: 'Scope', parent_unit : CompilationUnit) -> None:
        super().__init__(name)
        self.scope = Scope(parent_scope)
        self.compilation_unit = parent_unit.make_child(name.lower())
        # Functions are defined in the scope of their own bodies, allowing recursion.
        self.scope.add_symbol(self)

    def mcfunction_id(self):
        return f'{self.compilation_unit.namespace}:{self.compilation_unit.id}'
    
    def insert(self):
        return self.mcfunction_id()

class Scope:
    def __init__(self, parent : 'Scope' = None) -> None:
        self.parent = parent
        self.symbols : dict[str, Symbol] = dict()
        self.code : list[Command] = []
    
    def add_symbol(self, name : str, symbol : Symbol):
        if self.get_symbol(name):
            raise ParserError(f"Can't define symbol '{name}'; it's already been defined.")
        self.symbols[name] = symbol

    def get_symbol(self, name):
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent != None:
            return self.parent.get_symbol(name)
        else:
            return None
    
    def all_symbols(self) -> dict[str, Symbol]:
        result = self.symbols.copy()
        if self.parent != None:
            result = dict(result, **(self.parent.all_symbols()))
        return result;


#################################################
#                    Parsing                    #
#################################################


pattern_name = re.compile(r'[A-Za-z][A-Za-z0-9_]*')
pattern_string = re.compile(r'"(?:[^"\r\n]|\\")*"')
pattern_any_whitespace = re.compile(r'[^\S\r\n]*')
pattern_lparen  = re.compile(r'\(')
pattern_rparen  = re.compile(r'\)')
pattern_rcurly  = re.compile(r'\}')
pattern_rsquare = re.compile(r'\]')
pattern_comma   = re.compile(r'\,')
pattern_colon   = re.compile(r'\:')
pattern_directive_prefix = re.compile(f"{re.escape(syntax['directive_prefix'])}")
pattern_embed_escape = re.compile(r'\\')
pattern_embed_open         = re.compile(f"{re.escape(syntax['embed_open'])}")
pattern_embed_close        = re.compile(f"{re.escape(syntax['embed_close'])}")
pattern_not_newline        = re.compile(f".")

class Parser:
    def __init__(self, text : str) -> None:
        self.text = text
        self.index = 0

    def allow_strictly(self, pattern : re.Pattern):
        'Try to match a pattern, without ignoring leading whitespace.'
        match = pattern.match(self.text, self.index)
        if match:
            self.index = match.end(0)
            return match[0]
        else:
            return None
    
    def allow(self, pattern : re.Pattern, strict = False):
        global pattern_any_whitespace
        prev_index = self.index
        if not strict:
            self.allow_strictly(pattern_any_whitespace)
        result = self.allow_strictly(pattern)
        # Consume no tokens if the parse failed.
        if result==None:
            self.index = prev_index
        return result

    def expect(self, pattern : re.Pattern, strict = False):
        token = self.allow(pattern, strict)
        if token==None:
            raise ParserError(f"Expected pattern {pattern.pattern} but found --> {self.text[self.index:]}")
        return token
    
    def peek(self, pattern : re.Pattern = None):
        if pattern == None:
            return self.text[self.index] if self.any() else None
        else:
            match = pattern.match(self.text, self.index)
            if match:
                return match[0]
            else:
                return None

    def any(self):
        return self.index < len(self.text)
    
class FileParser:
    def __init__(self, lines : list[str]) -> None:
        self.index = -1
        self.lines = lines

    def next(self):
        self.index += 1
        return self.lines[self.index]
    
    def current(self):
        if self.index >= 0:
            return self.lines[self.index]
        else:
            return None
    
    def any(self):
        return self.index < len(self.lines)-1

def log_error(lineparser : FileParser, parser : Parser, err : ParserError):
    row = 0 if lineparser==None else lineparser.index + 1
    col = 0 if parser==None     else parser.index
    print(f'  {row},{col}: {str(err)}')
    if lineparser:
        print(f'  -->{lineparser.current()[col : col+40]}')

linenum = 0

def parse_expr(parser : Parser):
    global pattern_string
    global pattern_any_whitespace

    if parser.peek() == '"':
        return parser.expect(pattern_string, True)[1:-1]
    #else:
    #    return parser.expect(re.compile(r'[^,)\r\n]*'), True)

    pattern_outer_contents = re.compile(r'[^(){}\[\]"\r\n,]*')
    pattern_inner_contents = re.compile(r'[^(){}\[\]"\r\n]*')

    def parse_contents(contents_pattern : re.Pattern, end_char : str):
        result = ''
        while True:
            new = parser.allow(contents_pattern, True)
            if new==None or new=='':
                break
            result += new
            
            open_char = parser.allow(re.compile(r'\(|\{|\['))
            if open_char:
                result += open_char
                if open_char=='(':
                    result += parse_contents(pattern_inner_contents, ')')
                elif open_char=='[':
                    result += parse_contents(pattern_inner_contents, ']')
                elif open_char=='{':
                    result += parse_contents(pattern_inner_contents, '}')
            
            if parser.peek() == '"':
                result += parser.expect(pattern_string)
        
        if end_char:
            result += parser.expect(re.compile(fr'\{end_char}'))
        return result

    parser.allow(pattern_any_whitespace, True)
    return parse_contents(pattern_outer_contents, None)

def parse_sequence(parser : Parser, parse_item : Callable[[Parser], any], pattern_delimiter : re.Pattern):
    global pattern_any_whitespace
    
    result = []
    while True:
        parser.allow(pattern_any_whitespace, True)
        next = parse_item(parser)
        if next==None:
            break

        result.append(next)
        if not parser.allow(pattern_delimiter):
            break
    return result

class Embed():
    @staticmethod
    def parse(parser : Parser) -> 'Embed':
        parser.expect(pattern_embed_open)
        name = Code.parse(parser, re_union(pattern_embed_close, pattern_colon))

        result : Embed
        if parser.allow(pattern_colon):
            pattern_terminator = re_union(pattern_embed_close, pattern_comma)
            args = parse_sequence(parser, lambda p: Code.parse(p, pattern_terminator), pattern_comma)
            result = Invocation(name, args)
        else:
            result = Insertion(name)
        
        parser.expect(pattern_embed_close)

        return result
    
    def evaluate(self, scope : Scope) -> str:
        raise NotImplementedError()

class Insertion(Embed):
    def __init__(self, name : 'Code') -> None:
        super().__init__()
        self.name = name

    def evaluate(self, scope: Scope) -> str:
        name = self.name.evaluate(scope)
        symbol = scope.get_symbol(name)
        return symbol.insert()

class Invocation(Embed):
    def __init__(self, name : 'Code', args : list['Code']) -> None:
        super().__init__()
        self.name = name
        self.args = args
    
    def evaluate(self, scope: Scope) -> str:
        name = self.name.evaluate(scope)
        symbol = scope.get_symbol(name)
        args = [arg.evaluate(scope) for arg in self.args]
        return symbol.invoke(args)

class Code(list):
    @staticmethod
    def parse(parser : Parser, pattern_terminator : re.Pattern = None):
        result = Code()
        current_str = ''
        while parser.any():
            if pattern_terminator and parser.peek(pattern_terminator):
                break
            elif escape := parser.allow(pattern_embed_escape, strict=True):
                current_str += escape
                # Permit ONE protected token to follow the escape character.
                if text := parser.allow(pattern_embed_open, strict=True):
                    current_str += text
                elif text := parser.allow(pattern_embed_close, strict=True):
                    current_str += text
            elif parser.peek(pattern_embed_open, strict=True):
                if current_str:
                    result.append(current_str)
                    current_str = ''
                
                result.append(Embed.parse(parser))
            else:
                current_str += parser.expect(pattern_not_newline, strict=True)
        
        if current_str:
            result.append(current_str)
            current_str = ''
        
        return result

    def evaluate(self, scope : Scope) -> str:
        result = ''
        for item in self:
            if isinstance(item, str):
                result += item
            elif isinstance(item, Embed):
                result += item.evaluate(scope)
            else:
                raise RuntimeError(f"Code must contain only strs or Embeds, but contains: {repr(item)}")

class Command:
    def output(self, scope : Scope, lines : list[str]):
        raise NotImplementedError()

class CommandPlaintext(Command):
    def __init__(self, code : Code) -> None:
        self.code = code
    
    def output(self, scope : Scope, lines : list[str]):
        lines.append(self.code.evaluate(scope))

class CommandComment(Command):
    def __init__(self, text : str) -> None:
        self.text = text
    
    def output(self, scope : Scope, lines : list[str]):
        lines.append(self.text)

class CommandDefine(Command):
    def __init__(self, lineparser : FileParser, parser : Parser) -> None:
        global pattern_name
        global pattern_lparen
        global pattern_rparen
        global pattern_comma
        global pattern_any_whitespace

        self.name = parser.expect(pattern_name)

        self.arg_names = None
        if parser.allow(pattern_lparen):
            self.arg_names = parse_sequence(parser, lambda prsr: prsr.expect(pattern_name), pattern_comma)
            parser.expect(pattern_rparen)

        self.code = None
        self.commands = None
        parser.allow(pattern_any_whitespace, strict=True)
        if parser.any(): # Symbol is defined in 1 line:
            self.code = Code.parse(parser)
        else: # No inline definition:
            self.commands = parse_block(lineparser)
    
    def output(self, scope : Scope, lines : list[str]):
        symbol : Symbol
        if self.arg_names is None:
            # Define a variable. Variables are evaluated now, when they are defined.
            value : str
            if self.code:
                value = self.code.evaluate(scope)
            else:
                assert self.commands is not None

                inner_lines : list[str] = []
                for command in self.commands:
                    command.output(scope, inner_lines)
                value = '\n'.join(inner_lines)
            symbol = SymbolVariable(value)
        else:
            # Define a template. Templates are evaluated later, when they are invoked.
            commands : list[Command]
            if self.code:
                commands = [CommandPlaintext(self.code)]
            else:
                assert self.commands is not None
                commands = self.commands
            symbol = SymbolTemplate(scope, self.arg_names, commands)
        
        scope.add_symbol(self.name, symbol)

class CommandBlock(Command):
    do_not_inline = True

    def __init__(self, compilation_unit : CompilationUnit, lineparser : FileParser, scope : Scope, parser : Parser) -> None:
        self.condition_command = lineparser.next().strip(' \r\n\t')
        if not self.condition_command.removeprefix('$').startswith('execute '):
            raise ParserError(f"The line following a {syntax['directive_prefix']}block must be a stub 'execute' command.")
        self.scope = Scope(scope)
        self.scope.code = parse_block(lineparser)
        
        
    
    def output(self, scope: Scope, lines: list[str]):
        if CommandBlock.do_not_inline:
            self.compilation_unit = compilation_unit.make_child()

        else:
            # Prepend this block's condition to each command in the block. In other words, "inline" the block.
            for command in self.scope.code:
                if isinstance(command, CommandPlaintext):
                    line = command.code.evaluate(scope)
                    line = line.lstrip(' \r\n\t')
                    if not line.startswith('#'):
                        condition = self.condition_command
                        if line.startswith('execute'):
                            line = line.removeprefix('execute')
                        else:
                            line = f' run {line}'

                        if line.startswith('$'): # Macro line
                            line = line.removeprefix('$')
                            if not condition.startswith('$'):
                                condition = '$' + condition

                        line = condition + line
                    command.code = Code([line])

        inner_lines = []
        for command in self.scope.code:
            command.output(self.scope, inner_lines)

        compilation_unit_instance = self.compilation_unit.make_child()
        if CommandBlock.do_not_inline:
            write_output_file(inner_lines, compilation_unit_instance.path)
            CommandPlaintext(
                Code([ f'{self.condition_command} run function {compilation_unit_instance.namespace}:{compilation_unit_instance.id}\n' ])
            ).output(scope, lines)
        
        else:
            for line in inner_lines:
                lines.append(line)
        
class CommandFunction(Command):
    def __init__(self, compilation_unit : CompilationUnit, lineparser : FileParser, scope : Scope, parser : Parser) -> None:
        global pattern_name

        name = parser.expect(pattern_name)

        self.func = SymbolFunction(name, scope, compilation_unit)
        parse_block(self.func.compilation_unit, lineparser, self.func.scope)
        scope.add_symbol(name, self.func)

    def output(self, scope: Scope, lines: list[str]):
        inner_lines = []
        for command in self.func.scope.code:
            command.output(self.func.scope, inner_lines)
        write_output_file(inner_lines, self.func.compilation_unit.path)

class CommandImport(Command):
    def __init__(self, compilation_unit : CompilationUnit, scope : Scope, parser : Parser) -> None:
        global pattern_colon

        namespace = parser.expect(re.compile(r'[^:\r\n]+'))
        parser.expect(pattern_colon)
        name = parser.expect(re.compile(r'[^\r\n]+'))
        (imported_compilation_unit, imported_scope) = get_processed_file((namespace, name))
        imported_compilation_unit.dependents.add(compilation_unit)
        for name, symbol in imported_scope.all_symbols().items():
            try:
                scope.add_symbol(symbol)
            except ParserError as e:
                pass # Suppress errors from trying to add duplicate variables. TODO: Why?

class CommandEnd(Command):
    pass

def replace_symbols(scope : Scope, line : str) -> str:
    global pattern_lparen
    global pattern_comma

    symbols = scope.all_symbols().values()
    symbols.sort(key = lambda symbol: len(symbol.name), reverse=True)
    while True:
        for symbol in symbols:
            #match = re.compile(fr'\b{symbol.name}\b').search(line)
            match = re.compile(fr'{symbol.name}').search(line)
            if match:
                parser = Parser(line)
                parser.index = match.end()
                value = ''
                if isinstance(symbol, SymbolMacro):
                    arg_values = []
                    if parser.allow(pattern_lparen):
                        arg_values = parse_sequence(parser, parse_expr, pattern_comma)
                        parser.expect(pattern_rparen)
                    arg_values = [replace_symbols(scope, v) for v in arg_values]
                    lines = []
                    
                    try:
                        symbol.output(lines, arg_values)
                    except ParserError as err:
                        if err.pos == None:
                            err.pos = (0,0)
                        err.pos = (err.pos[1], parser.index+1)
                        raise err
                    
                    value = ''.join(lines).rstrip('\r\n')
                elif isinstance(symbol, SymbolArg):
                    value = symbol.value
                elif isinstance(symbol, SymbolFunction):
                    value = symbol.mcfunction_id()

                line = line[:match.start()] + value + line[parser.index:]
                break
        else:
            break
    return line

def parse_command(fileparser : FileParser) -> Command:
    line = fileparser.next()

    normalized_line = line.lstrip(' \r\n\t')
    parser = Parser(normalized_line)
    if parser.allow(pattern_directive_prefix): # Is preprocessor directive:
        opcode = parser.expect(pattern_name, strict=True)
        
        if opcode=='define':
            return CommandDefine(fileparser, parser)
        elif opcode=='block':
            return CommandBlock(compilation_unit, fileparser, scope, parser)
        elif opcode=='function':
            return CommandFunction(compilation_unit, fileparser, scope, parser)
        elif opcode=='import':
            return CommandImport(compilation_unit, scope, parser)
        elif opcode=='end':
            return CommandEnd()
        else:
            raise ParserError(f"Unrecognized opcode '{opcode}'")
    elif normalized_line.startswith('#'):
        return CommandComment(normalized_line)
    else:
        return CommandPlaintext(Code.parse(parser))

def parse_block(lineparser : FileParser) -> list[Command]:
    result = []
    while lineparser.any():
        command = parse_command(lineparser)
        if isinstance(command, CommandEnd):
            break
        result.append(command)
    else:
        raise ParserError(f"Expected closing {syntax['directive_prefix']}end, but found end-of-file.")

    return result




def write_output_file(lines : list[str], path : str):
    output_lines = [
        f'########################################################\n',
        f'###          TURBO PREPROCESSOR Output File          ###\n',
    #    f'###     Source:  {self.func.compilation_unit. ... parent?}\n',
        f'########################################################\n',
        f'\n',
    ] + lines

    output = ''
    for line in output_lines:
        output += line

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, '+w') as file:
        file.write(output)

id_to_processed_file : dict[tuple[str, str], tuple[CompilationUnit, Scope]] = dict()
def get_processed_file(id : tuple[str]):
    global id_to_processed_file

    if id not in id_to_processed_file:
        (namespace, name) = id
        input_path  = f'data/{namespace}/functions/{name}{syntax["src_extension"]}'
        output_path = f'data/{namespace}/functions/{name}{syntax["dest_extension"]}'
        if not os.path.exists(input_path):
            raise ParserError(f"Function {namespace}:{name} does not exist at path: {input_path}")

        compilation_unit = CompilationUnit(input_path, output_path)
        input_lines = []
        with open(input_path) as file:
            input_lines = file.readlines()
        lineparser = FileParser(input_lines)

        global_scope = Scope()
        try:
            commands = []
            while lineparser.any():
                commands.append(parse_command(lineparser))
        except ParserError as err:
            log_error(lineparser, None, err)
        
        id_to_processed_file[id] = (compilation_unit, global_scope)

    return id_to_processed_file[id]

def process(input_path):
    global cache

    print(f'Processing {input_path}...')
    id = path_to_id(input_path.removesuffix(syntax['src_extension']))
    (compilation_unit, global_scope) = get_processed_file(id)
    
    output_lines = [
        f'########################################################\n',
        f'###          TURBO PREPROCESSOR Output File          ###\n',
        f'###     Source:  {input_path}\n',
        f'########################################################\n',
        f'\n',
    ]
    for index, command in enumerate(global_scope.code):
        try:
            command.output(global_scope, output_lines)
        except ParserError as err:
            if err.pos == None:
                err.pos = (-1,-1)
            err.pos = (index+1, err.pos[1])
            log_error(None, None, err)

    output = ''
    for line in output_lines:
        output += line

    with open(compilation_unit.path, '+w') as file:
        file.write(output)
    cache['files'][input_path] = {
        'modified': os.path.getmtime(compilation_unit.path),
        'dependents': [],
    }

cache = {
    'version': cache_version,
    'files': {},
}

# Command line usage:
#   python turbo_preprocess.py [input_path]
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--syntax', required=False, help="The path of a .json file defining alternate source syntax.")
    argparser.add_argument('filename', nargs='?',      help="The path of a single source file to process.")
    args = argparser.parse_args()

    if args.syntax:
        with open(args.syntax, 'r') as syntax_file:
            alt_syntax = json.load(syntax_file)
            for key, value in alt_syntax.items():
                syntax[key] = value

    if args.filename:
        process(args.filename)
    else:
        # Load the cache from disk. Create a new cache if it's missing or using a different version.
        cachefilename = 'turbo_cache.json'
        if os.path.isfile(cachefilename):
            with open(cachefilename, 'r') as file:
                found_cache = json.load(file)
            found_cache_version = found_cache.get('version')
            if found_cache_version == cache_version:
                cache = found_cache
            else:
                print(f"Expected cache version {cache_version}, but found {found_cache_version}. Clearing the cache.")
            
        for dirpath, dirnames, filenames in os.walk('data'):
            for filename in filenames:
                if filename.endswith(syntax['src_extension']):
                    # If the file has changed, process it:
                    input_path = normalize_path(os.path.join(dirpath, filename))
                    if (input_path not in cache['files']) or os.path.getmtime(input_path) > cache['files'][input_path]['modified']:
                        process(input_path)
        
        # After processing files, update the cache.
        for compilation_unit, global_scope in id_to_processed_file.values():
            dependents = cache['files'][compilation_unit.source_path]['dependents']
            for dependent in compilation_unit.dependents:
                if dependent not in dependents:
                    dependents.append(dependent.source_path)

        with open(cachefilename, 'w+') as file:
            file.write(json.dumps(cache, indent='  '))

    print('\nDONE')