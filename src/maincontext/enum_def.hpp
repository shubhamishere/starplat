/*enum for various graph characteristics*/
#ifndef ENUMDEF_H
#define ENUMDEF_H
enum TYPE
{
  TYPE_LONG,
  TYPE_INT,
  TYPE_BOOL,
  TYPE_FLOAT,
  TYPE_DOUBLE,
  TYPE_STRING,
  TYPE_GRAPH,
  TYPE_GNN,
  TYPE_DIRGRAPH,
  TYPE_GEOMCOMPLETEGRAPH,
  TYPE_GRAPH_LIST,
  TYPE_LIST,
  TYPE_SETN,
  TYPE_SETE,
  TYPE_NODE,
  TYPE_EDGE,
  TYPE_PROPNODE,
  TYPE_PROPEDGE,
  TYPE_NONE,
  TYPE_UPDATES,
  TYPE_CONTAINER,
  TYPE_POINT,
  TYPE_UNDIREDGE,
  TYPE_TRIANGLE,
  TYPE_NODEMAP,
  TYPE_VECTOR,
  TYPE_SET,
  TYPE_HASHMAP,
  TYPE_HASHSET,
  TYPE_AUTOREF,
  TYPE_HEAP,
  TYPE_MAP,
  TYPE_BTREE,
};

inline bool check_isNodeEdgeType(int typeId)
{
  return ((typeId == TYPE_NODE) || (typeId == TYPE_EDGE));
}
inline bool check_isPropType(int typeId)
{
  return ((typeId == TYPE_PROPNODE) || (typeId == TYPE_PROPEDGE));
}
inline bool check_isIntegerType(int typeId)
{
  return ((typeId == TYPE_LONG) || (typeId == TYPE_INT));
}
inline bool check_isCollectionType(int typeId)
{
  return ((typeId == TYPE_LIST) || (typeId == TYPE_SETE) || (typeId == TYPE_SETN) || (typeId == TYPE_UPDATES) || (typeId == TYPE_NODEMAP) || (typeId == TYPE_CONTAINER) || (typeId == TYPE_VECTOR) || (typeId == TYPE_HASHMAP) || (typeId == TYPE_HASHSET) || (typeId == TYPE_BTREE)) || (typeId == TYPE_SET);
}
inline bool check_isGraphType(int typeId)
{
  return ((typeId == TYPE_GRAPH) || (typeId == TYPE_DIRGRAPH)) || (typeId == TYPE_GEOMCOMPLETEGRAPH) || (typeId == TYPE_GRAPH_LIST);
}
inline bool check_isGeomCompleteGraphType(int typeId)
{
  return (typeId==TYPE_GEOMCOMPLETEGRAPH);
}
inline bool check_isGNNType(int typeId)
{
  return ((typeId==TYPE_GNN));
}
inline bool check_isHeapType(int typeId)
{
  return typeId == TYPE_HEAP;
}
inline bool check_isMapType(int typeId)
{
  return typeId == TYPE_MAP;
}
inline bool check_isVectorType(int typeId)
{
  return typeId == TYPE_VECTOR;
}
inline bool check_isSetType(int typeId)
{
  return typeId == TYPE_SET;
}
inline bool check_isBTreeType(int typeId)
{
  return typeId == TYPE_BTREE;
}
inline bool check_isPrimitiveType(int typeId)
{
  return ((typeId == TYPE_BOOL) || (typeId == TYPE_DOUBLE) || (typeId == TYPE_FLOAT) || (typeId == TYPE_LONG) || (typeId == TYPE_INT) || (typeId == TYPE_AUTOREF)||(typeId == TYPE_STRING));
}

inline bool check_isPropNodeType(int typeId)
{
  return typeId == TYPE_PROPNODE;
}

inline bool check_isPropEdgeType(int typeId)
{
  return typeId == TYPE_PROPEDGE;
}
inline bool check_isListCollectionType(int typeId)
{
  return typeId == TYPE_LIST;
}
inline bool check_isSetCollectionType(int typeId)
{
  return ((typeId == TYPE_SETN) || (typeId == TYPE_SETE));
}
inline bool check_isNodeType(int typeId)
{
  return typeId == TYPE_NODE;
}
inline bool check_isEdgeType(int typeId)
{
  return typeId == TYPE_EDGE;
}
inline bool check_isContainerType(int typeId)
{

  return typeId == TYPE_CONTAINER;
}
inline bool check_isStructType(int typeId)
{

  return (typeId == TYPE_POINT || typeId == TYPE_UNDIREDGE || typeId == TYPE_TRIANGLE);
}
inline bool check_isPointType(int typeId)
{

  return typeId == TYPE_POINT;
}
inline bool check_isUndirectedEdgeType(int typeId)
{

  return typeId == TYPE_UNDIREDGE;
}
inline bool check_isTriangleType(int typeId)
{

  return typeId == TYPE_TRIANGLE;
}
inline bool check_isNodeMapType(int typeId)
{

  return typeId == TYPE_NODEMAP;
}
inline bool check_isHashMapType(int typeId)
{

  return typeId == TYPE_HASHMAP;
}
inline bool check_isHashSetType(int typeId)
{

  return typeId == TYPE_HASHSET;
}

enum REDUCE
{
  REDUCE_SUM,
  REDUCE_COUNT,
  REDUCE_PRODUCT,
  REDUCE_MAX,
  REDUCE_MIN,

};

enum OPERATOR
{
  OPERATOR_ADD,
  OPERATOR_SUB,
  OPERATOR_MUL,
  OPERATOR_DIV,
  OPERATOR_MOD,
  OPERATOR_OR,
  OPERATOR_AND,
  OPERATOR_LT,
  OPERATOR_GT,
  OPERATOR_LE,
  OPERATOR_GE,
  OPERATOR_EQ,
  OPERATOR_NE,
  OPERATOR_NOT,
  OPERATOR_INC,
  OPERATOR_DEC,
  OPERATOR_ADDASSIGN,
  OPERATOR_SUBASSIGN,
  OPERATOR_MULASSIGN,
  OPERATOR_DIVASSIGN,
  OPERATOR_ORASSIGN,
  OPERATOR_ANDASSIGN,
  OPERATOR_INDEX,

};

enum FUNCTYPE
{
  GEN_FUNC,
  STATIC_FUNC,
  INCREMENTAL_FUNC,
  DECREMENTAL_FUNC,
  DYNAMIC_FUNC,

};

enum NODETYPE
{
  NODE_ID,
  NODE_PROPACCESS,
  NODE_FUNC,
  NODE_TYPE,
  NODE_FORMALPARAM,
  NODE_STATEMENT,
  NODE_BLOCKSTMT,
  NODE_DECL,
  NODE_ASSIGN,
  NODE_WHILESTMT,
  NODE_SIMPLEFORSTMT,
  NODE_DOWHILESTMT,
  NODE_FIXEDPTSTMT,
  NODE_IFSTMT,
  NODE_ITRBFS,
  NODE_ITRRBFS,
  NODE_EXPR,
  NODE_PROCCALLEXPR,
  NODE_PROCCALLSTMT,
  NODE_FORALLSTMT,
  NODE_REDUCTIONCALL,
  NODE_REDUCTIONCALLSTMT,
  NODE_UNARYSTMT,
  NODE_RETURN,
  NODE_BATCHBLOCKSTMT,
  NODE_ONADDBLOCK,
  NODE_ONDELETEBLOCK,
  NODE_TRANSFERSTMT,
  NODE_LOOPSTMT,
  NODE_ALLOCATE,
  NODE_BREAKSTMT,
  NODE_CONTINUESTMT,
};

enum EXPR
{
  EXPR_RELATIONAL,
  EXPR_LOGICAL,
  EXPR_ARITHMETIC,
  EXPR_UNARY,
  EXPR_BOOLCONSTANT,
  EXPR_INTCONSTANT,
  EXPR_LONGCONSTANT,
  EXPR_DOUBLECONSTANT,
  EXPR_STRINGCONSTANT,
  EXPR_FLOATCONSTANT,
  EXPR_ID,
  EXPR_PROPID,
  EXPR_INFINITY,
  EXPR_PROCCALL,
  EXPR_DEPENDENT,
  EXPR_MAPGET,
  EXPR_ALLOCATE,
};

static const char *currentBatch = "currentBatch";
static const char *attachNodeCall = "attachNodeProperty";
static const char *attachEdgeCall = "attachEdgeProperty";
static const char *nbrCall = "neighbors";
static const char *edgeCall = "get_edge";
static const char *countOutNbrCall = "count_outNbrs";
static const char *isAnEdgeCall = "is_an_edge";
static const char *nodesToCall = "nodes_to";
static const char *nodesCall = "nodes";
static const char *getMSTCall = "getMST";
static const char *copyGraphCall = "copyGraph";
static const char *calculateDistanceCall = "calculateDistance";
static const char *makeGraphCopyCall = "makeGraphCopy";
static const char *getGraphAtIndexCall = "getGraphAtIndex";

#endif
