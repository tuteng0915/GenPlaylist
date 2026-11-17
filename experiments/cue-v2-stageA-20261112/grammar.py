"""Bounded JSON grammars prevent Qwen whitespace/string runaway under guidance."""
CHAR = r'char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})'
EXTRACT = r'''
root ::= "{\"candidates\": [" (candidate (", " candidate){0,23})? "]}"
candidate ::= "{\"phrase\": " phrase ", \"category\": " category ", \"source\": " source ", \"evidence\": " evidence ", \"confidence\": " confidence "}"
phrase ::= "\"" char{2,80} "\""
evidence ::= "\"" char{3,180} "\""
category ::= "\"sound\"" | "\"emotion\"" | "\"theme\""
source ::= "\"genre\"" | "\"tags\"" | "\"mood\"" | "\"lyrics\""
confidence ::= "0.0" | "0.25" | "0.5" | "0.6" | "0.7" | "0.75" | "0.8" | "0.85" | "0.9" | "0.95" | "1.0"
''' + CHAR

import json
literal=lambda s:json.dumps(s)
JUDGE='root ::= '+literal('{"judgments": [')+' '+(' '+literal(', ')+' ').join('j'+str(i) for i in range(8))+' '+literal(']}')+'\n'
for i in range(8):
 JUDGE+='j'+str(i)+' ::= '+literal('{"index": '+str(i)+', "category": ')+' category '+literal(', "supported": ')+' boolean '+literal(', "evidence": ')+' evidence '+literal(', "reason": ')+' reason '+literal('}')+'\n'
JUDGE+=r'''
category ::= "\"sound\"" | "\"emotion\"" | "\"theme\"" | "\"none\""
boolean ::= "true" | "false"
evidence ::= "\"" char{0,180} "\""
reason ::= "\"" char{0,240} "\""
'''+CHAR
