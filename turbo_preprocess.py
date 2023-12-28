import re
import os
import sys
import json
import argparse
from typing import Callable
from dataclasses import dataclass, field


"""
Turbo: A .mcfunction preprocessor
Turbo speeds up your .mcfunction coding with constructs including constants, macros, code blocks, and local functions.

by Silas Barber, 2023.

usage: turbo_preprocess.py [-h] [--syntax SYNTAX] [--verbose-parse] [filename]

positional arguments:
  filename         The path of a single source file to process.

optional arguments:
  -h, --help       show this help message and exit
  --syntax SYNTAX  The path of a .json file defining alternate source syntax.
  --verbose-parse  Print each syntax tree as it is parsed.
"""


#   TODO
# Allow the 'filename' arg to specify a source DIRECTORY, which defaults to ./data/ .
# Add an arg for a dest directory, which defaults to the source directory.
# Rebuild all when this python script changes.
# Test if working:  Rebuild dependee functions when a dependency function changes.
# Add --clean option.
# Clean output files for a compilation unit before outputting them. This prevents obsolete output files from persisting if they aren't overwritten.
# Maybe automatically clean output files for sources that no longer exist.

syntax = {
    'src_extension': ' SRC.mcfunction',
    'dest_extension': '.mcfunction',
    'directive_prefix': '##',
    'embed_open': '%(',
    'embed_close': ')',
}

cache_version = 2

def re_union(*patterns : re.Pattern):
    return re.compile('|'.join(compiled_pattern.pattern for compiled_pattern in patterns))

def normalize_path(path):
    return os.path.normpath(path).replace('\\', '/')

def path_to_id(path):
    path = normalize_path(path)
    relpath = normalize_path(os.path.relpath(path, 'data'))
    namespace = relpath.split('/', 1)[0]
    name = os.path.splitext(relpath)[0].removeprefix(f'{namespace}/functions/')
    return (namespace, name)

class TurboFunction:
    def __init__(self, path : str) -> None:
        self.path = path
        (self.namespace, self.name) = path_to_id(path.removesuffix(syntax['src_extension']))
        self.dependents : set[TurboFunction] = set()
        self.commands : list[Command] = []

    def id(self):
        return f'{self.namespace}:{self.id}'

class MinecraftFunction:
    def __init__(self, path : str, source : TurboFunction, parent : 'MinecraftFunction' = None) -> None:
        self.path = path
        (self.namespace, self.name) = path_to_id(path.removesuffix(syntax['dest_extension']))
        self.source = source
        self.parent = parent
        self.anonymous_child_count = 0
        'Used to give numbered names to anonymous child functions.'
        self.lines : list[str] = []

    def id(self):
        return f'{self.namespace}:{self.name}'
    
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
        return MinecraftFunction(path, source=self.source, parent=self)
    
    def write(self):
        output = ''
        for line in self.lines:
            output += line
    
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, '+w') as file:
            file.write(output)

def output_file_header(target : MinecraftFunction):
    return [
        f'########################################################\n',
        f'###          TURBO PREPROCESSOR Output File          ###\n',
        f'###     Source:  {target.source.path}\n',
        f'########################################################\n',
        f'\n',
    ]

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

class SymbolVariable(Symbol):
    def __init__(self, value : str) -> None:
        self.value = value

    def insert(self) -> str:
        return self.value

class SymbolTemplate(Symbol):
    def __init__(self, parent_scope : 'Scope', parent_target : MinecraftFunction, argnames : list[str], commands : list['Command']) -> None:
        self.parent_scope = parent_scope
        self.parent_target = parent_target
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

        inner_target = self.parent_target.make_child()
        for command in self.commands:
            command.output(inner_scope, inner_target)
        return '\n'.join(inner_target.lines)

class SymbolFunction(Symbol):
    def __init__(self, id: str) -> None:
        self.id = id
    
    def insert(self):
        return self.id

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
        
    def require_symbol(self, name):
        symbol = self.get_symbol(name)
        if symbol is None:
            raise ParserError(f"No such symbol: {name}")
        return symbol
    
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
pattern_newline = re.compile(r"\r?\n")
pattern_lparen  = re.compile(r'\(')
pattern_rparen  = re.compile(r'\)')
pattern_rcurly  = re.compile(r'\}')
pattern_rsquare = re.compile(r'\]')
pattern_comma   = re.compile(r'\,')
pattern_colon   = re.compile(r'\:')
pattern_quote   = re.compile(r'`')
pattern_directive_prefix = re.compile(f"{re.escape(syntax['directive_prefix'])}")
pattern_embed_escape = re.compile(r'\\')
pattern_embed_open         = re.compile(f"{re.escape(syntax['embed_open'])}")
pattern_embed_close        = re.compile(f"{re.escape(syntax['embed_close'])}")
pattern_not_newline = re.compile(f".")

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
        name = ArgCode.parse(parser, re_union(pattern_embed_close, pattern_colon))

        result : Embed
        if parser.allow(pattern_colon):
            pattern_terminator = re_union(pattern_embed_close, pattern_comma)
            args = parse_sequence(parser, lambda p: ArgCode.parse(p, pattern_terminator), pattern_comma)
            result = Invocation(name, args)
        else:
            result = Insertion(name)
        
        parser.expect(pattern_embed_close)

        return result
    
    def evaluate(self, scope : Scope):
        raise NotImplementedError()

class Insertion(Embed):
    def __init__(self, name : 'Code') -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"(Insertion: {repr(self.name)})"

    def evaluate(self, scope: Scope):
        name = self.name.evaluate(scope)
        symbol = scope.require_symbol(name)
        return symbol.insert()

class Invocation(Embed):
    def __init__(self, name : 'Code', args : list['Code']) -> None:
        super().__init__()
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        return f"(Invocation: {repr(self.name)}, {repr(self.args)})"
    
    def evaluate(self, scope: Scope):
        name = self.name.evaluate(scope)
        symbol = scope.require_symbol(name)
        args = [arg.evaluate(scope) for arg in self.args]
        return symbol.invoke(args)

class Code(list):
    @staticmethod
    def parse(parser : Parser, pattern_terminator : re.Pattern):
        result = Code()
        current_str = ''
        while parser.any():
            if pattern_terminator and parser.peek(pattern_terminator):
                break
            elif escape := parser.allow(pattern_embed_escape, strict=True):
                protected_token_patterns = (pattern_embed_open, pattern_terminator)
                if text := next((match for match in (parser.allow(pattern, strict=True) for pattern in protected_token_patterns) if match), None):
                    # Permit ONE protected token to follow the escape character.
                    current_str += text
                elif text := parser.allow(pattern_embed_escape, strict=True):
                    # "\\" -> "\" only if followed by a protected token. Otherwise, "\\" -> "\\".
                    if any(parser.peek(pattern) for pattern in protected_token_patterns):
                        current_str += text
                    else:
                        current_str += escape + text
                else:
                    # If no protected token follows the escape, transclude the escape.
                    current_str += escape
            elif parser.peek(pattern_embed_open):
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

    def evaluate(self, scope : Scope):
        result = ''
        for item in self:
            evaluated = None
            if isinstance(item, str):
                evaluated = item
            elif isinstance(item, Embed):
                evaluated = item.evaluate(scope)
            else:
                raise RuntimeError(f"Code must contain only strs or Embeds, but contains: {repr(item)}")

            if isinstance(evaluated, str):
                if not isinstance(result, str):
                    raise ParserError(f"Can't mix a symbol embed with other code, but found: {repr(evaluated)} after: {repr(result)}")
                result += evaluated
            else:
                assert isinstance(evaluated, Symbol)
                if result != '':
                    raise ParserError(f"Can't mix a symbol embed with other code, but found: {repr(evaluated)} after: {repr(result)}")
                result = evaluated

        return result

class ArgCode(Code):
    @staticmethod
    def parse(parser : Parser, pattern_terminator : re.Pattern):
        result : str
        # Allow quoted argument code, which includes surrounding whitespace.
        if parser.allow(pattern_quote):
            result = Code.parse(parser, pattern_quote)

            parser.expect(pattern_quote)
        else:
            result = Code.parse(parser, pattern_terminator)
        
            # Strip surrounding whitespace.
            if len(result) > 0:
                if isinstance(result[0], str):
                    result[0] = result[0].lstrip()
                if isinstance(result[-1], str):
                    result[-1] = result[-1].rstrip()
        
        return result

class Command:
    def output(self, scope : Scope, target : MinecraftFunction):
        raise NotImplementedError()

class CommandPlaintext(Command):
    def __init__(self, code : Code) -> None:
        self.code = code

    def __repr__(self) -> str:
        return f"(CommandPlaintext: {repr(self.code)})"

    @staticmethod
    def parse(parser : Parser) -> 'CommandPlaintext':
        code = Code.parse(parser, pattern_newline)
        code.append(parser.allow(pattern_newline, strict=True) or '')
        return CommandPlaintext(code)

    def output(self, scope : Scope, target : MinecraftFunction):
        evaluated = self.code.evaluate(scope)
        if isinstance(evaluated, Symbol):
            raise ParserError(f"A symbol embed is not a valid command, but found: {repr(evaluated)}")
        target.lines.append(evaluated)

class CommandComment(Command):
    def __init__(self, text : str) -> None:
        self.text = text
    
    def output(self, scope : Scope, target : MinecraftFunction):
        target.lines.append(self.text)

class CommandDefine(Command):
    def __init__(self, lineparser : FileParser, parser : Parser) -> None:

        self.name = parser.expect(pattern_name)

        self.arg_names = None
        if parser.allow(pattern_lparen):
            self.arg_names = parse_sequence(parser, lambda prsr: prsr.expect(pattern_name), pattern_comma)
            parser.expect(pattern_rparen)

        self.code = None
        self.commands = None
        parser.allow(pattern_any_whitespace, strict=True)
        if parser.any(): # Symbol is defined in 1 line:
            self.code = ArgCode.parse(parser, pattern_newline)
            parser.allow(pattern_newline, strict=True)
        else: # No inline definition:
            self.commands = parse_block(lineparser)
    
    def output(self, scope : Scope, target : MinecraftFunction):
        symbol : Symbol
        if self.arg_names is None:
            # Define a variable. Variables are evaluated now, when they are defined.
            value : str
            if self.code is not None:
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
            if self.code is not None:
                commands = [CommandPlaintext(self.code)]
            else:
                assert self.commands is not None
                commands = self.commands
            symbol = SymbolTemplate(scope, target, self.arg_names, commands)
        
        scope.add_symbol(self.name, symbol)

class CommandBlock(Command):
    do_not_inline = True

    def __init__(self, lineparser : FileParser, parser : Parser) -> None:
        parser.allow(pattern_any_whitespace, strict=True)
        if parser.any():
            raise ParserError(f"Expected nothing after {syntax['directive_prefix']}block but found: {parser.peek()}")
        
        self.condition_command = lineparser.next().strip(' \r\n\t')
        if not self.condition_command.removeprefix('$').startswith('execute '):
            raise ParserError(f"The line following a {syntax['directive_prefix']}block must be a stub 'execute' command.")
        self.commands = parse_block(lineparser)
    
    def output(self, scope: Scope, target : MinecraftFunction):
        if CommandBlock.do_not_inline:
            pass

        else:
            # Prepend this block's condition to each command in the block. In other words, "inline" the block.
            for command in self.commands:
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

        inner_scope = Scope(scope)

        if CommandBlock.do_not_inline:
            inner_target = target.make_child()
            inner_target.lines = output_file_header(inner_target)
            for command in self.commands:
                command.output(inner_scope, inner_target)
            inner_target.write()

            CommandPlaintext(
                Code([ f'{self.condition_command} run function {inner_target.id()}\n' ])
            ).output(scope, target)
        else:
            for command in self.commands:
                command.output(inner_scope, target)
        
class CommandFunction(Command):
    def __init__(self, lineparser : FileParser, parser : Parser) -> None:
        self.name = parser.expect(pattern_name)
        parser.allow(pattern_any_whitespace, strict=True)
        if parser.any() and not parser.peek(pattern_newline):
            raise ParserError(f"Expected nothing after the function name, but found: {repr(parser.peek())}")
        self.commands = parse_block(lineparser)

    def output(self, scope: Scope, target : MinecraftFunction):
        inner_target = target.make_child(self.name.lower())
        inner_target.lines = output_file_header(inner_target)
        symbol = SymbolFunction(inner_target.id())
        inner_scope = Scope(scope)
        # Functions are defined in the scope of their own bodies, allowing recursion.
        scope.add_symbol(self.name, symbol)
        for command in self.commands:
            command.output(inner_scope, inner_target)

        inner_target.write()

class CommandImport(Command):
    def __init__(self, parser : Parser) -> None:
        global pattern_colon

        self.namespace = parser.expect(re.compile(r'[^:\r\n]+'))
        parser.expect(pattern_colon)
        self.name = parser.expect(re.compile(r'[^\r\n]+'))
    
    def output(self, scope: Scope, target : MinecraftFunction):
        (imported_compilation_unit, imported_scope) = get_processed_file((self.namespace, self.name))
        imported_compilation_unit.dependents.add(source)
        for name, symbol in imported_scope.all_symbols().items():
            try:
                scope.add_symbol(name, symbol)
            except ParserError as e:
                pass # Suppress errors from trying to add duplicate variables. TODO: Why?

class CommandEnd(Command):
    pass

def parse_command(fileparser : FileParser) -> Command:
    line = fileparser.next()

    normalized_line = line.lstrip(' \r\n\t')
    parser = Parser(normalized_line)
    if parser.allow(pattern_directive_prefix): # Is preprocessor directive:
        opcode = parser.expect(pattern_name, strict=True)
        
        if opcode=='define':
            return CommandDefine(fileparser, parser)
        elif opcode=='block':
            return CommandBlock(fileparser, parser)
        elif opcode=='function':
            return CommandFunction(fileparser, parser)
        elif opcode=='import':
            return CommandImport(parser)
        elif opcode=='end':
            return CommandEnd()
        else:
            raise ParserError(f"Unrecognized opcode '{opcode}'")
    elif normalized_line.startswith('#'):
        return CommandComment(normalized_line)
    else:
        return CommandPlaintext.parse(parser)

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


##################################################
#                    Building                    #
##################################################


id_to_processed_file : dict[tuple[str, str], tuple[TurboFunction, Scope]] = dict()
def get_processed_file(id : tuple[str]):
    global id_to_processed_file

    if id not in id_to_processed_file:
        (namespace, name) = id
        input_path  = f'data/{namespace}/functions/{name}{syntax["src_extension"]}'
        output_path = f'data/{namespace}/functions/{name}{syntax["dest_extension"]}'
        if not os.path.exists(input_path):
            raise ParserError(f"Function {namespace}:{name} does not exist at path: {input_path}")

        source = TurboFunction(input_path)
        input_lines = []
        with open(input_path) as file:
            input_lines = file.readlines()
        lineparser = FileParser(input_lines)

        try:
            while lineparser.any():
                source.commands.append(parse_command(lineparser))
        except ParserError as err:
            log_error(lineparser, None, err)
        
        if verbose_parse:
            for command in source.commands:
                print(repr(command))

        global_scope = Scope()

        
        id_to_processed_file[id] = (source, global_scope)

    return id_to_processed_file[id]

def process(input_path, output_path):
    global cache

    print(f'Processing {input_path}...')
    # Imports can cause sources to be processed before they're iterated over in the source directory. get_processed_file() handles this.
    id = path_to_id(input_path.removesuffix(syntax['src_extension']))
    (source, global_scope) = get_processed_file(id)
    
    target = MinecraftFunction(output_path, source)
    target.lines = output_file_header(target)

    for index, command in enumerate(source.commands):
        try:
            command.output(global_scope, target)
        except ParserError as err:
            if err.pos == None:
                err.pos = (-1,-1)
            err.pos = (index+1, err.pos[1])
            log_error(None, None, err)

    target.write()
    
    # Update the cache.
    cache_source = cache['sources'].get(input_path, CacheSource())
    cache['sources'][input_path] = cache_source
    cache_artifact = cache_source['artifacts'].get(target.path, CacheArtifact())
    cache_source['artifacts'][target.path] = cache_artifact
    cache_artifact['modified'] = os.path.getmtime(target.path)

def CacheArtifact():
    return {
        'modified': -1.0,
    }

def CacheSource():
    return {
        'dependents': list(),
        'artifacts': dict(),
    }

def Cache():
    return {
        'version': cache_version,
        'sources': dict(),
    }

cache = Cache()
verbose_parse : bool = False

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--syntax', required=False, help="The path of a .json file defining alternate source syntax.")
    argparser.add_argument('--verbose-parse', required=False, action='store_true', help="Print each syntax tree as it is parsed. Defaults to True if 'source' is a file.")
    argparser.add_argument('--rebuild', required=False, action='store_true', help="Process each source file, even if it is up-to-date.")
    argparser.add_argument('source',      nargs='?', default='data/', help="The path to a source file or directory containing source files.")
    argparser.add_argument('destination', nargs='?',                  help="The path under which to output .mcfunction files, or the path at which to create a single .mcfunction file. Defaults to outputting alongside source files.")
    args = argparser.parse_args()

    verbose_parse = args.verbose_parse
    source_dest_are_dirs = os.path.isdir(args.source)
    destination : str
    if args.destination is not None:
        destination = args.destination
    else:
        if source_dest_are_dirs:
            destination = args.source
        else:
            destination = args.source.removesuffix(syntax['src_extension']) + syntax['dest_extension']

    if source_dest_are_dirs and not os.path.isdir(destination):
        raise ValueError(f"When 'source' is a directory like {args.source}, 'destination' must also be an existing directory, but instead it was: {destination}")

    if args.syntax:
        with open(args.syntax, 'r') as syntax_file:
            alt_syntax = json.load(syntax_file)
            for key, value in alt_syntax.items():
                syntax[key] = value

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
    
    # Iterate over the source files, and process them if needed.
    def source_paths_in(dir):
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if filename.endswith(syntax['src_extension']):
                    yield os.path.join(dirpath, filename)

    source_paths = source_paths_in(args.source) if source_dest_are_dirs else (args.source,)
    for source_path in source_paths:
        # If the file has changed, process it:
        input_path = normalize_path(os.path.relpath(source_path, '.'))
        output_path : str
        if source_dest_are_dirs:
            input_relpath = os.path.relpath(source_path, args.source)
            output_path = destination + '/' + input_relpath.removesuffix(syntax['src_extension']) + syntax['dest_extension']
            output_path = normalize_path(os.path.relpath(output_path, '.'))
        else: # destination is file:
            output_path = normalize_path(os.path.relpath(destination, '.'))

        is_dirty = (
            (input_path not in cache['sources'])
            or (output_path not in cache['sources'][input_path]['artifacts'])
            or os.path.getmtime(input_path) > cache['sources'][input_path]['artifacts'][output_path]['modified']
        )
        if args.rebuild or is_dirty:
            process(input_path, output_path)
    
    # After processing files, update dependencies in the cache.
    for source, global_scope in id_to_processed_file.values():
        dependents = cache['sources'][source.path]['dependents']
        for dependent in source.dependents:
            if dependent not in dependents:
                dependents.append(dependent.path)

    with open(cachefilename, 'w+') as file:
        file.write(json.dumps(cache, indent='  '))

    print('\nDONE')