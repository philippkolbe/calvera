Traverse all edges of a node:
- PUSH_MARK
- # do something for current edge
- NEXT_NODE
- IF_IS_NOT_FIRST_EDGE:
    - JUMP
- POP_MARK

Traverse all nodes and edges:

- PUSH_FIRST_NODE
- PUSH_MARK
- # do something current node
- NEXT_NODE
- IF_IS_NOT_FIRST_NODE:
    - JUMP
- POP_MARK