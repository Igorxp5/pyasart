from typing import Dict, Tuple

from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token, _TokenType

DEFAULT_CODE_TOKEN_COLOR = (111, 126, 127)

_CODE_TOKEN_COLOR = {
    Token.Comment: (80, 86, 80),
    Token.Keyword: (0, 191, 223),
    Token.String: (191, 192, 0),
    Token.Number: (128, 128, 239),
    Token.Name: (111, 126, 127),
    Token.Name.Function: (128, 191, 128),
    Token.Operator: (191, 191, 224),
}


def get_tokens_style() -> Dict[_TokenType, Tuple[int, int, int]]:
    styles = dict()
    for token, rgb in _CODE_TOKEN_COLOR.items():
        styles[token] = rgb
        for subtype in token.subtypes:
            styles[subtype] = rgb
    return styles


def get_code_token_colors(code: str):
    lexer = PythonLexer()

    styles = get_tokens_style()

    token_colors = []
    for token, value in lex(code, lexer):
        color = styles.get(token) or DEFAULT_CODE_TOKEN_COLOR
        token_colors.append((value, color))

    return token_colors
