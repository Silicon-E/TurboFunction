import re
import os
import sys
import json
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


def path_to_id(path):
    path = path.replace('\\', '/')
    relpath = os.path.relpath(path, 'data').replace('\\', '/')
    namespace = relpath.split('/', 1)[0]
    name = os.path.splitext(relpath)[0].removeprefix(f'{namespace}/functions/')
    return (namespace, name)

class CompilationUnit:
    def __init__(self, path : str, parent : 'CompilationUnit' = None) -> None:
        self.path = path
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
        path = self.path.removesuffix('.mcfunction')
        # If the parent unit has no parent (it is top-level), put the new child unit in a __turbo folder alongside the parent unit.
        # Otherwise, put it in the same directory as the parent.
        if not self.parent:
            head, tail = os.path.split(path)
            path = f'{head}/__turbo/{tail}'
        path += f'_{name}.mcfunction'
        return CompilationUnit(path, parent=self)

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

class SymbolArg(Symbol):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.value = None

class SymbolMacro(Symbol):
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

class SymbolFunction(Symbol):
    def __init__(self, name: str, parent_scope: 'Scope', parent_unit : CompilationUnit) -> None:
        super().__init__(name)
        self.scope = Scope(parent_scope)
        self.compilation_unit = parent_unit.make_child(name.lower())
        # Functions are defined in the scope of their own bodies, allowing recursion.
        self.scope.add_symbol(self)

    def mcfunction_id(self):
        return f'{self.compilation_unit.namespace}:{self.compilation_unit.id}'

class Scope:
    def __init__(self, parent : 'Scope' = None) -> None:
        self.parent = parent
        self.symbols : dict[str, Symbol] = dict()
        self.code : list[Command] = []
    
    def add_symbol(self, symbol : Symbol):
        if self.get_symbol(symbol.name):
            raise ParserError(f"Can't define symbol '{symbol.name}'; it's already been defined.")
        self.symbols[symbol.name] = symbol

    def get_symbol(self, name):
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent != None:
            return self.parent.get_symbol(name)
        else:
            return None
    
    def all_symbols(self) -> list[Symbol]:
        result = [symbol for symbol in self.symbols.values()]
        if self.parent != None:
            result += self.parent.all_symbols()
        return result;

pattern_name = re.compile(r'[A-Z][A-Z0-9_]*')
pattern_string = re.compile(r'"(?:[^"\r\n]|\\")*"')
pattern_any_whitespace = re.compile(r'[^\S\r\n]*')
pattern_lparen  = re.compile(r'\(')
pattern_rparen  = re.compile(r'\)')
pattern_rcurly  = re.compile(r'\}')
pattern_rsquare = re.compile(r'\]')
pattern_comma   = re.compile(r'\,')
pattern_colon   = re.compile(r'\:')

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

class Command:
    def output(self, scope : Scope, lines : list[str]):
        pass

class CommandPlaintext(Command):
    def __init__(self, text : str) -> None:
        self.text = text
    
    def output(self, scope : Scope, lines : list[str]):
        lines.append(replace_symbols(scope, self.text))

class CommandComment(Command):
    def __init__(self, text : str) -> None:
        self.text = text
    
    def output(self, scope : Scope, lines : list[str]):
        lines.append(self.text)

class CommandDefine(Command):
    def __init__(self, compilation_unit : CompilationUnit, lineparser : FileParser, scope : Scope, parser : Parser) -> None:
        global pattern_name
        global pattern_lparen
        global pattern_rparen
        global pattern_comma
        global pattern_any_whitespace

        name = parser.expect(pattern_name)
        arg_names = []
        if parser.allow(pattern_lparen):
            arg_names = parse_sequence(parser, lambda prsr: prsr.expect(pattern_name), pattern_comma)
            parser.expect(pattern_rparen)

        symbol = SymbolMacro(name, arg_names, scope)
        parser.allow(pattern_any_whitespace, True)
        if parser.any(): # Symbol is defined in 1 line:
            symbol.scope.code.append(CommandPlaintext(parse_expr(parser)))
        else: # No inline definition:
            parse_block(compilation_unit, lineparser, symbol.scope)
        scope.add_symbol(symbol)

class CommandInsert(Command):
    def __init__(self, scope : Scope, parser : Parser) -> None:
        global pattern_name
        global pattern_lparen
        global pattern_rparen
        global pattern_comma

        name = parser.expect(pattern_name)
        self.arg_values = []
        if parser.allow(pattern_lparen):
            self.arg_values = parse_sequence(parser, parse_expr, pattern_comma)
            parser.expect(pattern_rparen)

        symbol = scope.get_symbol(name)
        if symbol==None:
            raise ParserError(f"Used symbol '{name}', but it hasn't been defined yet!")
        if not isinstance(symbol, SymbolMacro):
            raise ParserError(f"Inserted symbol '{name}' must be a macro defined with ##define.")
        self.symbol = symbol
    
    def output(self, scope : Scope, lines : list[str]):
        self.symbol.output(lines, self.arg_values)

class CommandBlock(Command):
    do_not_inline = True

    def __init__(self, compilation_unit : CompilationUnit, lineparser : FileParser, scope : Scope, parser : Parser) -> None:
        self.condition_command = lineparser.next().strip(' \r\n\t')
        if not self.condition_command.startswith('execute '):
            raise ParserError(f"The line following a ##block must be a stub 'execute' command.")
        self.scope = Scope(scope)
        
        if CommandBlock.do_not_inline:
            self.compilation_unit = compilation_unit.make_child()
            parse_block(self.compilation_unit, lineparser, self.scope)

        else:
            parse_block(compilation_unit, lineparser, self.scope)

            # Prepend this block's condition to each command in the block. In other words, "inline" the block.
            for command in self.scope.code:
                if isinstance(command, CommandPlaintext):
                    line = command.text
                    line = line.lstrip(' \r\n\t')
                    if not line.startswith('#'):
                        if line.startswith('execute'):
                            line = line.removeprefix('execute')
                        else:
                            line = f' run {line}'
                        line = self.condition_command + line
                    command.text = line
    
    def output(self, scope: Scope, lines: list[str]):
        inner_lines = []
        for command in self.scope.code:
            command.output(self.scope, inner_lines)

        compilation_unit_instance = self.compilation_unit.make_child()
        if CommandBlock.do_not_inline:
            write_output_file(inner_lines, compilation_unit_instance.path)
            CommandPlaintext(
                f'{self.condition_command} run function {compilation_unit_instance.namespace}:{compilation_unit_instance.id}\n'
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
        scope.add_symbol(self.func)

    def output(self, scope: Scope, lines: list[str]):
        inner_lines = []
        for command in self.func.scope.code:
            command.output(self.func.scope, inner_lines)
        write_output_file(inner_lines, self.func.compilation_unit.path)

class CommandImport(Command):
    def __init__(self, scope : Scope, parser : Parser) -> None:
        global pattern_colon

        namespace = parser.expect(re.compile(r'[^:\r\n]+'))
        parser.expect(pattern_colon)
        name = parser.expect(re.compile(r'[^\r\n]+'))
        (compilation_unit, global_scope) = get_processed_file((namespace, name))
        for symbol in global_scope.all_symbols():
            try:
                scope.add_symbol(symbol)
            except ParserError as e:
                pass # Suppress errors from trying to add duplicate variables.

class CommandEnd(Command):
    pass

def replace_symbols(scope : Scope, line : str) -> str:
    global pattern_lparen
    global pattern_comma

    symbols = scope.all_symbols()
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

def parse_command(compilation_unit : CompilationUnit, fileparser : FileParser, scope : Scope) -> Command:
    'Returns True if the parsed line is a ##end directive, or False otherwise.'
    line = fileparser.next()

    normalized_line = line.lstrip(' \r\n\t')
    pattern_directive = re.compile(r'^##[^#]')
    if pattern_directive.search(normalized_line): # Is preprocessor directive:
        directive = normalized_line.removeprefix('##').removesuffix('\n')
        opcode = re.match(re.compile(r'\S*'), directive)[0]
        operand = directive.removeprefix(opcode).lstrip(' \r\n\t')
        parser = Parser(operand)

        if opcode=='define':
            return CommandDefine(compilation_unit, fileparser, scope, parser)
        elif opcode=='insert':
            return CommandInsert(scope, parser)
        elif opcode=='block':
            return CommandBlock(compilation_unit, fileparser, scope, parser)
        elif opcode=='function':
            return CommandFunction(compilation_unit, fileparser, scope, parser)
        elif opcode=='import':
            return CommandImport(scope, parser)
        elif opcode=='end':
            return CommandEnd()
        else:
            raise ParserError(f"Unrecognized opcode '{opcode}'")
    elif normalized_line.startswith('#'):
        return CommandComment(normalized_line)
    else:
        return CommandPlaintext(normalized_line)

def parse_block(compilation_unit : CompilationUnit, lineparser : FileParser, scope : Scope):
    while lineparser.any():
        command = parse_command(compilation_unit, lineparser, scope)
        scope.code.append(command)
        if isinstance(command, CommandEnd):
            break




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
def get_processed_file(id):
    global id_to_processed_file

    if id not in id_to_processed_file:
        (namespace, name) = id
        input_path  = f'data/{namespace}/functions/{name} SRC.mcfunction'
        output_path = f'data/{namespace}/functions/{name}.mcfunction'
        if not os.path.exists(input_path):
            raise ParserError(f"Function {namespace}:{name} does not exist at path: {input_path}")

        compilation_unit = CompilationUnit(output_path)
        input_lines = []
        with open(input_path) as file:
            input_lines = file.readlines()
        lineparser = FileParser(input_lines)

        global_scope = Scope()
        try:
            parse_block(compilation_unit, lineparser, global_scope)
        except ParserError as err:
            log_error(lineparser, None, err)
        
        id_to_processed_file[id] = (compilation_unit, global_scope)

    return id_to_processed_file[id]

path_to_modifytime = {}

def process(input_path):
    global path_to_modifytime

    print(f'Processing {input_path}...')
    id = path_to_id(input_path.removesuffix(' SRC.mcfunction'))
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
    path_to_modifytime[input_path] = os.path.getmtime(compilation_unit.path)

# Command line usage:
#   python turbo_preprocess.py [input_path]
if __name__ == '__main__':
    if len(sys.argv) >= 2:
        process(sys.argv[1])
    else:
        path_to_modifytime = {}
        cachefilename = 'turbo_cache.json'
        if os.path.isfile(cachefilename):
            with open(cachefilename, 'r') as file:
                path_to_modifytime = json.load(file)

        for root, dirnames, filenames in os.walk('data'):
            for filename in filenames:
                if filename.endswith(' SRC.mcfunction'):
                    # If the file has changed, process it:
                    input_path = os.path.join(root, filename)
                    if (input_path not in path_to_modifytime) or os.path.getmtime(input_path) > path_to_modifytime[input_path]:
                        process(input_path)
        
        with open(cachefilename, 'w+') as file:
            file.write(json.dumps(path_to_modifytime, indent='  '))

    print('\nDONE')