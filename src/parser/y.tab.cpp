/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 2 "lrparser.y"

	#include <stdio.h>
	#include <string.h>
	#include <stdlib.h>
	#include <stdbool.h>
    #include "includeHeader.hpp"
	#include "../analyser/attachProp/attachPropAnalyser.h"
	#include "../analyser/dataRace/dataRaceAnalyser.h"
	#include "../analyser/deviceVars/deviceVarsAnalyser.h"
	#include "../analyser/pushpull/pushpullAnalyser.h"
	#include "../analyser/callGraph/callGraphAnalyser.h"
	#include "../analyser/blockVars/blockVarsAnalyser.h"
	#include "../analyser/cudaGlobalVariables/cudaGlobalVariablesAnalyser.h"
	#include<getopt.h>
	//#include "../symbolutil/SymbolTableBuilder.cpp"
     
	void yyerror(const char *);
	int yylex(void);
    extern FILE* yyin;

	char mytext[100];
	char var[100];
	int num = 0;
	vector<map<int,vector<Identifier*>>> graphId(5);
	extern char *yytext;
	//extern SymbolTable* symbTab;
	FrontEndContext frontEndContext;
	map<string,int> push_map;
	set<string> allGpuUsedVars;
	char* backendTarget ;
    vector<Identifier*> tempIds; //stores graph vars in current function's param list.
    //symbTab=new SymbolTable();
	//symbolTableList.push_back(new SymbolTable());

#line 106 "y.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    T_INT = 258,                   /* T_INT  */
    T_FLOAT = 259,                 /* T_FLOAT  */
    T_BOOL = 260,                  /* T_BOOL  */
    T_DOUBLE = 261,                /* T_DOUBLE  */
    T_STRING = 262,                /* T_STRING  */
    T_LONG = 263,                  /* T_LONG  */
    T_AUTOREF = 264,               /* T_AUTOREF  */
    T_FORALL = 265,                /* T_FORALL  */
    T_FOR = 266,                   /* T_FOR  */
    T_P_INF = 267,                 /* T_P_INF  */
    T_INF = 268,                   /* T_INF  */
    T_N_INF = 269,                 /* T_N_INF  */
    T_LOOP = 270,                  /* T_LOOP  */
    T_FUNC = 271,                  /* T_FUNC  */
    T_IF = 272,                    /* T_IF  */
    T_ELSE = 273,                  /* T_ELSE  */
    T_WHILE = 274,                 /* T_WHILE  */
    T_RETURN = 275,                /* T_RETURN  */
    T_DO = 276,                    /* T_DO  */
    T_IN = 277,                    /* T_IN  */
    T_FIXEDPOINT = 278,            /* T_FIXEDPOINT  */
    T_UNTIL = 279,                 /* T_UNTIL  */
    T_FILTER = 280,                /* T_FILTER  */
    T_TO = 281,                    /* T_TO  */
    T_BY = 282,                    /* T_BY  */
    T_ADD_ASSIGN = 283,            /* T_ADD_ASSIGN  */
    T_SUB_ASSIGN = 284,            /* T_SUB_ASSIGN  */
    T_MUL_ASSIGN = 285,            /* T_MUL_ASSIGN  */
    T_DIV_ASSIGN = 286,            /* T_DIV_ASSIGN  */
    T_MOD_ASSIGN = 287,            /* T_MOD_ASSIGN  */
    T_AND_ASSIGN = 288,            /* T_AND_ASSIGN  */
    T_XOR_ASSIGN = 289,            /* T_XOR_ASSIGN  */
    T_OR_ASSIGN = 290,             /* T_OR_ASSIGN  */
    T_INC_OP = 291,                /* T_INC_OP  */
    T_DEC_OP = 292,                /* T_DEC_OP  */
    T_PTR_OP = 293,                /* T_PTR_OP  */
    T_AND_OP = 294,                /* T_AND_OP  */
    T_OR_OP = 295,                 /* T_OR_OP  */
    T_LE_OP = 296,                 /* T_LE_OP  */
    T_GE_OP = 297,                 /* T_GE_OP  */
    T_EQ_OP = 298,                 /* T_EQ_OP  */
    T_NE_OP = 299,                 /* T_NE_OP  */
    T_ASTERISK = 300,              /* T_ASTERISK  */
    T_AND = 301,                   /* T_AND  */
    T_OR = 302,                    /* T_OR  */
    T_SUM = 303,                   /* T_SUM  */
    T_AVG = 304,                   /* T_AVG  */
    T_COUNT = 305,                 /* T_COUNT  */
    T_PRODUCT = 306,               /* T_PRODUCT  */
    T_MAX = 307,                   /* T_MAX  */
    T_MIN = 308,                   /* T_MIN  */
    T_GRAPH = 309,                 /* T_GRAPH  */
    T_GNN = 310,                   /* T_GNN  */
    T_DIR_GRAPH = 311,             /* T_DIR_GRAPH  */
    T_NODE = 312,                  /* T_NODE  */
    T_EDGE = 313,                  /* T_EDGE  */
    T_UPDATES = 314,               /* T_UPDATES  */
    T_CONTAINER = 315,             /* T_CONTAINER  */
    T_POINT = 316,                 /* T_POINT  */
    T_UNDIREDGE = 317,             /* T_UNDIREDGE  */
    T_TRIANGLE = 318,              /* T_TRIANGLE  */
    T_NODEMAP = 319,               /* T_NODEMAP  */
    T_VECTOR = 320,                /* T_VECTOR  */
    T_HASHMAP = 321,               /* T_HASHMAP  */
    T_HASHSET = 322,               /* T_HASHSET  */
    T_BTREE = 323,                 /* T_BTREE  */
    T_GEOMCOMPLETEGRAPH = 324,     /* T_GEOMCOMPLETEGRAPH  */
    T_GRAPH_LIST = 325,            /* T_GRAPH_LIST  */
    T_SET = 326,                   /* T_SET  */
    T_NP = 327,                    /* T_NP  */
    T_EP = 328,                    /* T_EP  */
    T_LIST = 329,                  /* T_LIST  */
    T_SET_NODES = 330,             /* T_SET_NODES  */
    T_SET_EDGES = 331,             /* T_SET_EDGES  */
    T_FROM = 332,                  /* T_FROM  */
    T_RANDOMSHUFFLE = 333,         /* T_RANDOMSHUFFLE  */
    T_ALLOCATE = 334,              /* T_ALLOCATE  */
    T_BREAK = 335,                 /* T_BREAK  */
    T_CONTINUE = 336,              /* T_CONTINUE  */
    T_BFS = 337,                   /* T_BFS  */
    T_REVERSE = 338,               /* T_REVERSE  */
    T_INCREMENTAL = 339,           /* T_INCREMENTAL  */
    T_DECREMENTAL = 340,           /* T_DECREMENTAL  */
    T_STATIC = 341,                /* T_STATIC  */
    T_DYNAMIC = 342,               /* T_DYNAMIC  */
    T_BATCH = 343,                 /* T_BATCH  */
    T_ONADD = 344,                 /* T_ONADD  */
    T_ONDELETE = 345,              /* T_ONDELETE  */
    return_func = 346,             /* return_func  */
    ID = 347,                      /* ID  */
    INT_NUM = 348,                 /* INT_NUM  */
    FLOAT_NUM = 349,               /* FLOAT_NUM  */
    BOOL_VAL = 350,                /* BOOL_VAL  */
    CHAR_VAL = 351,                /* CHAR_VAL  */
    STRING_VAL = 352               /* STRING_VAL  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif
/* Token kinds.  */
#define YYEMPTY -2
#define YYEOF 0
#define YYerror 256
#define YYUNDEF 257
#define T_INT 258
#define T_FLOAT 259
#define T_BOOL 260
#define T_DOUBLE 261
#define T_STRING 262
#define T_LONG 263
#define T_AUTOREF 264
#define T_FORALL 265
#define T_FOR 266
#define T_P_INF 267
#define T_INF 268
#define T_N_INF 269
#define T_LOOP 270
#define T_FUNC 271
#define T_IF 272
#define T_ELSE 273
#define T_WHILE 274
#define T_RETURN 275
#define T_DO 276
#define T_IN 277
#define T_FIXEDPOINT 278
#define T_UNTIL 279
#define T_FILTER 280
#define T_TO 281
#define T_BY 282
#define T_ADD_ASSIGN 283
#define T_SUB_ASSIGN 284
#define T_MUL_ASSIGN 285
#define T_DIV_ASSIGN 286
#define T_MOD_ASSIGN 287
#define T_AND_ASSIGN 288
#define T_XOR_ASSIGN 289
#define T_OR_ASSIGN 290
#define T_INC_OP 291
#define T_DEC_OP 292
#define T_PTR_OP 293
#define T_AND_OP 294
#define T_OR_OP 295
#define T_LE_OP 296
#define T_GE_OP 297
#define T_EQ_OP 298
#define T_NE_OP 299
#define T_ASTERISK 300
#define T_AND 301
#define T_OR 302
#define T_SUM 303
#define T_AVG 304
#define T_COUNT 305
#define T_PRODUCT 306
#define T_MAX 307
#define T_MIN 308
#define T_GRAPH 309
#define T_GNN 310
#define T_DIR_GRAPH 311
#define T_NODE 312
#define T_EDGE 313
#define T_UPDATES 314
#define T_CONTAINER 315
#define T_POINT 316
#define T_UNDIREDGE 317
#define T_TRIANGLE 318
#define T_NODEMAP 319
#define T_VECTOR 320
#define T_HASHMAP 321
#define T_HASHSET 322
#define T_BTREE 323
#define T_GEOMCOMPLETEGRAPH 324
#define T_GRAPH_LIST 325
#define T_SET 326
#define T_NP 327
#define T_EP 328
#define T_LIST 329
#define T_SET_NODES 330
#define T_SET_EDGES 331
#define T_FROM 332
#define T_RANDOMSHUFFLE 333
#define T_ALLOCATE 334
#define T_BREAK 335
#define T_CONTINUE 336
#define T_BFS 337
#define T_REVERSE 338
#define T_INCREMENTAL 339
#define T_DECREMENTAL 340
#define T_STATIC 341
#define T_DYNAMIC 342
#define T_BATCH 343
#define T_ONADD 344
#define T_ONDELETE 345
#define return_func 346
#define ID 347
#define INT_NUM 348
#define FLOAT_NUM 349
#define BOOL_VAL 350
#define CHAR_VAL 351
#define STRING_VAL 352

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 40 "lrparser.y"

    int  info;
    long ival;
	bool bval;
    double fval;
    char* text;
	char cval;
	ASTNode* node;
	paramList* pList;
	argList* aList;
	ASTNodeList* nodeList;
    tempNode* temporary;
     

#line 368 "y.tab.c"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_T_INT = 3,                      /* T_INT  */
  YYSYMBOL_T_FLOAT = 4,                    /* T_FLOAT  */
  YYSYMBOL_T_BOOL = 5,                     /* T_BOOL  */
  YYSYMBOL_T_DOUBLE = 6,                   /* T_DOUBLE  */
  YYSYMBOL_T_STRING = 7,                   /* T_STRING  */
  YYSYMBOL_T_LONG = 8,                     /* T_LONG  */
  YYSYMBOL_T_AUTOREF = 9,                  /* T_AUTOREF  */
  YYSYMBOL_T_FORALL = 10,                  /* T_FORALL  */
  YYSYMBOL_T_FOR = 11,                     /* T_FOR  */
  YYSYMBOL_T_P_INF = 12,                   /* T_P_INF  */
  YYSYMBOL_T_INF = 13,                     /* T_INF  */
  YYSYMBOL_T_N_INF = 14,                   /* T_N_INF  */
  YYSYMBOL_T_LOOP = 15,                    /* T_LOOP  */
  YYSYMBOL_T_FUNC = 16,                    /* T_FUNC  */
  YYSYMBOL_T_IF = 17,                      /* T_IF  */
  YYSYMBOL_T_ELSE = 18,                    /* T_ELSE  */
  YYSYMBOL_T_WHILE = 19,                   /* T_WHILE  */
  YYSYMBOL_T_RETURN = 20,                  /* T_RETURN  */
  YYSYMBOL_T_DO = 21,                      /* T_DO  */
  YYSYMBOL_T_IN = 22,                      /* T_IN  */
  YYSYMBOL_T_FIXEDPOINT = 23,              /* T_FIXEDPOINT  */
  YYSYMBOL_T_UNTIL = 24,                   /* T_UNTIL  */
  YYSYMBOL_T_FILTER = 25,                  /* T_FILTER  */
  YYSYMBOL_T_TO = 26,                      /* T_TO  */
  YYSYMBOL_T_BY = 27,                      /* T_BY  */
  YYSYMBOL_T_ADD_ASSIGN = 28,              /* T_ADD_ASSIGN  */
  YYSYMBOL_T_SUB_ASSIGN = 29,              /* T_SUB_ASSIGN  */
  YYSYMBOL_T_MUL_ASSIGN = 30,              /* T_MUL_ASSIGN  */
  YYSYMBOL_T_DIV_ASSIGN = 31,              /* T_DIV_ASSIGN  */
  YYSYMBOL_T_MOD_ASSIGN = 32,              /* T_MOD_ASSIGN  */
  YYSYMBOL_T_AND_ASSIGN = 33,              /* T_AND_ASSIGN  */
  YYSYMBOL_T_XOR_ASSIGN = 34,              /* T_XOR_ASSIGN  */
  YYSYMBOL_T_OR_ASSIGN = 35,               /* T_OR_ASSIGN  */
  YYSYMBOL_T_INC_OP = 36,                  /* T_INC_OP  */
  YYSYMBOL_T_DEC_OP = 37,                  /* T_DEC_OP  */
  YYSYMBOL_T_PTR_OP = 38,                  /* T_PTR_OP  */
  YYSYMBOL_T_AND_OP = 39,                  /* T_AND_OP  */
  YYSYMBOL_T_OR_OP = 40,                   /* T_OR_OP  */
  YYSYMBOL_T_LE_OP = 41,                   /* T_LE_OP  */
  YYSYMBOL_T_GE_OP = 42,                   /* T_GE_OP  */
  YYSYMBOL_T_EQ_OP = 43,                   /* T_EQ_OP  */
  YYSYMBOL_T_NE_OP = 44,                   /* T_NE_OP  */
  YYSYMBOL_T_ASTERISK = 45,                /* T_ASTERISK  */
  YYSYMBOL_T_AND = 46,                     /* T_AND  */
  YYSYMBOL_T_OR = 47,                      /* T_OR  */
  YYSYMBOL_T_SUM = 48,                     /* T_SUM  */
  YYSYMBOL_T_AVG = 49,                     /* T_AVG  */
  YYSYMBOL_T_COUNT = 50,                   /* T_COUNT  */
  YYSYMBOL_T_PRODUCT = 51,                 /* T_PRODUCT  */
  YYSYMBOL_T_MAX = 52,                     /* T_MAX  */
  YYSYMBOL_T_MIN = 53,                     /* T_MIN  */
  YYSYMBOL_T_GRAPH = 54,                   /* T_GRAPH  */
  YYSYMBOL_T_GNN = 55,                     /* T_GNN  */
  YYSYMBOL_T_DIR_GRAPH = 56,               /* T_DIR_GRAPH  */
  YYSYMBOL_T_NODE = 57,                    /* T_NODE  */
  YYSYMBOL_T_EDGE = 58,                    /* T_EDGE  */
  YYSYMBOL_T_UPDATES = 59,                 /* T_UPDATES  */
  YYSYMBOL_T_CONTAINER = 60,               /* T_CONTAINER  */
  YYSYMBOL_T_POINT = 61,                   /* T_POINT  */
  YYSYMBOL_T_UNDIREDGE = 62,               /* T_UNDIREDGE  */
  YYSYMBOL_T_TRIANGLE = 63,                /* T_TRIANGLE  */
  YYSYMBOL_T_NODEMAP = 64,                 /* T_NODEMAP  */
  YYSYMBOL_T_VECTOR = 65,                  /* T_VECTOR  */
  YYSYMBOL_T_HASHMAP = 66,                 /* T_HASHMAP  */
  YYSYMBOL_T_HASHSET = 67,                 /* T_HASHSET  */
  YYSYMBOL_T_BTREE = 68,                   /* T_BTREE  */
  YYSYMBOL_T_GEOMCOMPLETEGRAPH = 69,       /* T_GEOMCOMPLETEGRAPH  */
  YYSYMBOL_T_GRAPH_LIST = 70,              /* T_GRAPH_LIST  */
  YYSYMBOL_T_SET = 71,                     /* T_SET  */
  YYSYMBOL_T_NP = 72,                      /* T_NP  */
  YYSYMBOL_T_EP = 73,                      /* T_EP  */
  YYSYMBOL_T_LIST = 74,                    /* T_LIST  */
  YYSYMBOL_T_SET_NODES = 75,               /* T_SET_NODES  */
  YYSYMBOL_T_SET_EDGES = 76,               /* T_SET_EDGES  */
  YYSYMBOL_T_FROM = 77,                    /* T_FROM  */
  YYSYMBOL_T_RANDOMSHUFFLE = 78,           /* T_RANDOMSHUFFLE  */
  YYSYMBOL_T_ALLOCATE = 79,                /* T_ALLOCATE  */
  YYSYMBOL_T_BREAK = 80,                   /* T_BREAK  */
  YYSYMBOL_T_CONTINUE = 81,                /* T_CONTINUE  */
  YYSYMBOL_T_BFS = 82,                     /* T_BFS  */
  YYSYMBOL_T_REVERSE = 83,                 /* T_REVERSE  */
  YYSYMBOL_T_INCREMENTAL = 84,             /* T_INCREMENTAL  */
  YYSYMBOL_T_DECREMENTAL = 85,             /* T_DECREMENTAL  */
  YYSYMBOL_T_STATIC = 86,                  /* T_STATIC  */
  YYSYMBOL_T_DYNAMIC = 87,                 /* T_DYNAMIC  */
  YYSYMBOL_T_BATCH = 88,                   /* T_BATCH  */
  YYSYMBOL_T_ONADD = 89,                   /* T_ONADD  */
  YYSYMBOL_T_ONDELETE = 90,                /* T_ONDELETE  */
  YYSYMBOL_return_func = 91,               /* return_func  */
  YYSYMBOL_ID = 92,                        /* ID  */
  YYSYMBOL_INT_NUM = 93,                   /* INT_NUM  */
  YYSYMBOL_FLOAT_NUM = 94,                 /* FLOAT_NUM  */
  YYSYMBOL_BOOL_VAL = 95,                  /* BOOL_VAL  */
  YYSYMBOL_CHAR_VAL = 96,                  /* CHAR_VAL  */
  YYSYMBOL_STRING_VAL = 97,                /* STRING_VAL  */
  YYSYMBOL_98_ = 98,                       /* '?'  */
  YYSYMBOL_99_ = 99,                       /* ':'  */
  YYSYMBOL_100_ = 100,                     /* '<'  */
  YYSYMBOL_101_ = 101,                     /* '>'  */
  YYSYMBOL_102_ = 102,                     /* '+'  */
  YYSYMBOL_103_ = 103,                     /* '-'  */
  YYSYMBOL_104_ = 104,                     /* '*'  */
  YYSYMBOL_105_ = 105,                     /* '/'  */
  YYSYMBOL_106_ = 106,                     /* '%'  */
  YYSYMBOL_107_ = 107,                     /* '('  */
  YYSYMBOL_108_ = 108,                     /* ')'  */
  YYSYMBOL_109_ = 109,                     /* ','  */
  YYSYMBOL_110_ = 110,                     /* '&'  */
  YYSYMBOL_111_ = 111,                     /* ';'  */
  YYSYMBOL_112_ = 112,                     /* '.'  */
  YYSYMBOL_113_ = 113,                     /* '{'  */
  YYSYMBOL_114_ = 114,                     /* '}'  */
  YYSYMBOL_115_ = 115,                     /* '='  */
  YYSYMBOL_116_ = 116,                     /* '!'  */
  YYSYMBOL_117_ = 117,                     /* '['  */
  YYSYMBOL_118_ = 118,                     /* ']'  */
  YYSYMBOL_YYACCEPT = 119,                 /* $accept  */
  YYSYMBOL_program = 120,                  /* program  */
  YYSYMBOL_function_def = 121,             /* function_def  */
  YYSYMBOL_function_data = 122,            /* function_data  */
  YYSYMBOL_paramList = 123,                /* paramList  */
  YYSYMBOL_type = 124,                     /* type  */
  YYSYMBOL_param = 125,                    /* param  */
  YYSYMBOL_by_reference = 126,             /* by_reference  */
  YYSYMBOL_function_body = 127,            /* function_body  */
  YYSYMBOL_statements = 128,               /* statements  */
  YYSYMBOL_statement = 129,                /* statement  */
  YYSYMBOL_break_stmt = 130,               /* break_stmt  */
  YYSYMBOL_continue_stmt = 131,            /* continue_stmt  */
  YYSYMBOL_blockstatements = 132,          /* blockstatements  */
  YYSYMBOL_batch_blockstmt = 133,          /* batch_blockstmt  */
  YYSYMBOL_on_add_blockstmt = 134,         /* on_add_blockstmt  */
  YYSYMBOL_on_delete_blockstmt = 135,      /* on_delete_blockstmt  */
  YYSYMBOL_block_begin = 136,              /* block_begin  */
  YYSYMBOL_block_end = 137,                /* block_end  */
  YYSYMBOL_return_stmt = 138,              /* return_stmt  */
  YYSYMBOL_declaration = 139,              /* declaration  */
  YYSYMBOL_type1 = 140,                    /* type1  */
  YYSYMBOL_primitive = 141,                /* primitive  */
  YYSYMBOL_type3 = 142,                    /* type3  */
  YYSYMBOL_graph = 143,                    /* graph  */
  YYSYMBOL_gnn = 144,                      /* gnn  */
  YYSYMBOL_collections = 145,              /* collections  */
  YYSYMBOL_structs = 146,                  /* structs  */
  YYSYMBOL_container = 147,                /* container  */
  YYSYMBOL_vector = 148,                   /* vector  */
  YYSYMBOL_set = 149,                      /* set  */
  YYSYMBOL_nodemap = 150,                  /* nodemap  */
  YYSYMBOL_hashmap = 151,                  /* hashmap  */
  YYSYMBOL_hashset = 152,                  /* hashset  */
  YYSYMBOL_btree = 153,                    /* btree  */
  YYSYMBOL_type2 = 154,                    /* type2  */
  YYSYMBOL_property = 155,                 /* property  */
  YYSYMBOL_assignment = 156,               /* assignment  */
  YYSYMBOL_rhs = 157,                      /* rhs  */
  YYSYMBOL_expression = 158,               /* expression  */
  YYSYMBOL_alloca_expr = 159,              /* alloca_expr  */
  YYSYMBOL_indexExpr = 160,                /* indexExpr  */
  YYSYMBOL_unary_expr = 161,               /* unary_expr  */
  YYSYMBOL_proc_call = 162,                /* proc_call  */
  YYSYMBOL_val = 163,                      /* val  */
  YYSYMBOL_control_flow = 164,             /* control_flow  */
  YYSYMBOL_iteration_cf = 165,             /* iteration_cf  */
  YYSYMBOL_filterExpr = 166,               /* filterExpr  */
  YYSYMBOL_boolean_expr = 167,             /* boolean_expr  */
  YYSYMBOL_selection_cf = 168,             /* selection_cf  */
  YYSYMBOL_reduction = 169,                /* reduction  */
  YYSYMBOL_reduce_op = 170,                /* reduce_op  */
  YYSYMBOL_leftList = 171,                 /* leftList  */
  YYSYMBOL_rightList = 172,                /* rightList  */
  YYSYMBOL_reductionCall = 173,            /* reductionCall  */
  YYSYMBOL_reduction_calls = 174,          /* reduction_calls  */
  YYSYMBOL_leftSide = 175,                 /* leftSide  */
  YYSYMBOL_arg_list = 176,                 /* arg_list  */
  YYSYMBOL_bfs_abstraction = 177,          /* bfs_abstraction  */
  YYSYMBOL_reverse_abstraction = 178,      /* reverse_abstraction  */
  YYSYMBOL_oid = 179,                      /* oid  */
  YYSYMBOL_tid = 180,                      /* tid  */
  YYSYMBOL_id = 181                        /* id  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1398

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  119
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  63
/* YYNRULES -- Number of rules.  */
#define YYNRULES  208
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  488

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   352


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   116,     2,     2,     2,   106,   110,     2,
     107,   108,   104,   102,   109,   103,   112,   105,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    99,   111,
     100,   115,   101,    98,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   117,     2,   118,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   113,     2,   114,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   109,   109,   110,   112,   119,   125,   131,   137,   143,
     150,   151,   154,   155,   156,   158,   175,   179,   186,   199,
     200,   206,   209,   210,   212,   213,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   227,   228,
     230,   232,   234,   236,   238,   240,   242,   243,   246,   254,
     257,   260,   263,   267,   271,   275,   280,   281,   282,   283,
     284,   289,   292,   293,   294,   295,   296,   297,   299,   301,
     302,   303,   304,   307,   309,   310,   312,   314,   315,   316,
     317,   318,   319,   320,   321,   323,   324,   325,   327,   328,
     329,   332,   333,   334,   336,   339,   340,   341,   343,   346,
     348,   351,   354,   356,   357,   358,   360,   361,   362,   363,
     364,   366,   369,   370,   371,   374,   376,   377,   378,   379,
     380,   381,   382,   383,   384,   385,   386,   387,   388,   389,
     390,   391,   392,   395,   396,   397,   398,   399,   401,   405,
     407,   408,   410,   421,   425,   429,   440,   441,   442,   443,
     444,   445,   446,   449,   450,   452,   453,   454,   455,   457,
     458,   459,   460,   461,   462,   463,   465,   466,   468,   470,
     471,   474,   475,   477,   478,   481,   482,   483,   484,   485,
     487,   489,   491,   492,   494,   495,   504,   506,   507,   508,
     509,   510,   512,   513,   514,   515,   518,   522,   539,   546,
     551,   559,   560,   564,   565,   568,   572,   579,   583
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "T_INT", "T_FLOAT",
  "T_BOOL", "T_DOUBLE", "T_STRING", "T_LONG", "T_AUTOREF", "T_FORALL",
  "T_FOR", "T_P_INF", "T_INF", "T_N_INF", "T_LOOP", "T_FUNC", "T_IF",
  "T_ELSE", "T_WHILE", "T_RETURN", "T_DO", "T_IN", "T_FIXEDPOINT",
  "T_UNTIL", "T_FILTER", "T_TO", "T_BY", "T_ADD_ASSIGN", "T_SUB_ASSIGN",
  "T_MUL_ASSIGN", "T_DIV_ASSIGN", "T_MOD_ASSIGN", "T_AND_ASSIGN",
  "T_XOR_ASSIGN", "T_OR_ASSIGN", "T_INC_OP", "T_DEC_OP", "T_PTR_OP",
  "T_AND_OP", "T_OR_OP", "T_LE_OP", "T_GE_OP", "T_EQ_OP", "T_NE_OP",
  "T_ASTERISK", "T_AND", "T_OR", "T_SUM", "T_AVG", "T_COUNT", "T_PRODUCT",
  "T_MAX", "T_MIN", "T_GRAPH", "T_GNN", "T_DIR_GRAPH", "T_NODE", "T_EDGE",
  "T_UPDATES", "T_CONTAINER", "T_POINT", "T_UNDIREDGE", "T_TRIANGLE",
  "T_NODEMAP", "T_VECTOR", "T_HASHMAP", "T_HASHSET", "T_BTREE",
  "T_GEOMCOMPLETEGRAPH", "T_GRAPH_LIST", "T_SET", "T_NP", "T_EP", "T_LIST",
  "T_SET_NODES", "T_SET_EDGES", "T_FROM", "T_RANDOMSHUFFLE", "T_ALLOCATE",
  "T_BREAK", "T_CONTINUE", "T_BFS", "T_REVERSE", "T_INCREMENTAL",
  "T_DECREMENTAL", "T_STATIC", "T_DYNAMIC", "T_BATCH", "T_ONADD",
  "T_ONDELETE", "return_func", "ID", "INT_NUM", "FLOAT_NUM", "BOOL_VAL",
  "CHAR_VAL", "STRING_VAL", "'?'", "':'", "'<'", "'>'", "'+'", "'-'",
  "'*'", "'/'", "'%'", "'('", "')'", "','", "'&'", "';'", "'.'", "'{'",
  "'}'", "'='", "'!'", "'['", "']'", "$accept", "program", "function_def",
  "function_data", "paramList", "type", "param", "by_reference",
  "function_body", "statements", "statement", "break_stmt",
  "continue_stmt", "blockstatements", "batch_blockstmt",
  "on_add_blockstmt", "on_delete_blockstmt", "block_begin", "block_end",
  "return_stmt", "declaration", "type1", "primitive", "type3", "graph",
  "gnn", "collections", "structs", "container", "vector", "set", "nodemap",
  "hashmap", "hashset", "btree", "type2", "property", "assignment", "rhs",
  "expression", "alloca_expr", "indexExpr", "unary_expr", "proc_call",
  "val", "control_flow", "iteration_cf", "filterExpr", "boolean_expr",
  "selection_cf", "reduction", "reduce_op", "leftList", "rightList",
  "reductionCall", "reduction_calls", "leftSide", "arg_list",
  "bfs_abstraction", "reverse_abstraction", "oid", "tid", "id", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-408)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-208)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -408,    57,  -408,   -42,   -32,   -27,   -42,   -42,  -408,    27,
    -408,   -16,  1228,  1228,    23,    39,  -408,  -408,  -408,  -408,
    1228,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,  -408,    50,    59,  -408,  -408,  -408,    60,    70,    73,
      88,  -408,  -408,  -408,    97,   113,   118,  -408,   123,   126,
     125,   122,   -36,  -408,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,  -408,   124,  -408,   127,  1228,  1228,
     728,   136,   -42,  1205,  1205,  1205,  1205,  1205,  1205,  1248,
    1322,   -42,   -42,  -408,  1228,  -408,   -42,   -42,  -408,   -42,
    -408,   137,   139,  -408,   146,   152,  -408,  -408,  -408,   155,
     156,   159,    11,    27,   246,   172,  -408,  -408,   170,   171,
     173,   176,   177,   179,  -408,  -408,  -408,  -408,   135,   135,
    -408,   135,  -408,   168,   181,  -408,  -408,  -408,  -408,  -408,
     182,   184,    -8,   -42,   -42,   185,   653,  -408,     2,   191,
     192,  -408,  -408,  -408,  -408,   194,    30,  -408,  -408,  -408,
     -86,  -408,   186,   187,   261,  -408,  -408,   201,   209,   202,
     211,   212,   213,   214,   215,   216,   222,   229,   230,   231,
    -408,  -408,  -408,   232,  -408,  -408,   -42,    28,   -42,   135,
     135,  -408,  1080,   -85,  -408,  -408,   233,   226,   325,   238,
    1205,   -42,   135,   135,   -42,   -42,   -42,  1080,   250,    44,
     939,  1080,  -408,  -408,  -408,  -408,   -22,     9,    16,  -408,
    -408,  -408,  -408,  -408,  -408,  -408,  -408,   135,   135,   135,
     135,   135,   135,   135,   135,   135,   135,   135,   135,   135,
     135,   135,   135,   135,   135,  -408,  -408,  -408,   135,   517,
     135,   -42,   135,  -408,   240,  -408,     3,  1205,  -408,     5,
    -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,   -42,   331,
     -42,   336,   337,  1080,   258,   260,   264,   -42,   271,   351,
     267,   949,    21,   269,   270,   280,   358,   361,   273,   135,
    -408,   135,   135,   135,   135,   135,   135,  1100,  1090,   359,
     359,   433,   433,  1080,   359,   359,   144,   144,     4,     4,
       4,   920,  1080,   278,  -408,  1080,   281,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,   285,  1080,   -51,  1080,   135,   135,
    -408,   292,   135,  -408,   286,   135,   283,   135,   135,   842,
      27,   135,   301,   298,   -42,   135,   135,   135,  -408,  -408,
     135,   -42,   -42,   306,  -408,   300,  -408,   302,  -408,   304,
    -408,  -408,   135,  -408,   135,   -42,   -42,   -70,    92,  -408,
     103,  -408,    75,    -7,   303,   135,    91,   108,   305,   472,
     395,  -408,   308,   135,   135,   307,  -408,  -408,   959,   309,
     310,   204,  -408,  -408,  -408,   315,   316,  -408,   296,  -408,
    1205,  -408,  1205,  -408,  1205,    27,    27,   135,   314,    27,
      27,   135,   135,   842,   317,   969,   318,   135,    27,   135,
     135,   320,   128,  -408,  -408,   319,   322,   323,  -408,  -408,
      17,   -10,   135,  -408,  -408,    17,   608,  -408,  -408,    27,
    -408,   341,  -408,   324,   326,   135,  -408,  -408,  -408,   408,
     327,   -42,   328,   329,   135,  -408,   -42,   339,   346,   340,
     335,    99,   333,    27,    54,   135,    27,   991,   344,    27,
      27,   135,  -408,   135,   135,  -408,  1070,  -408,    27,   342,
    -408,  -408,  -408,  -408,   345,    27,  -408,    27,  -408,  -408,
     365,  -105,  -408,   135,  -408,   349,    27,  -408
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     1,     0,     0,     0,     0,     0,     3,     0,
     208,     0,     0,     0,     0,     0,    44,     4,    21,    22,
       0,    62,    63,    64,    65,    67,    66,    69,    73,    70,
     103,   104,     0,     0,    85,    86,    87,     0,     0,     0,
       0,   102,    71,    72,     0,     0,     0,    74,     0,     0,
       0,    10,    19,    56,    57,    61,    58,    59,    78,    79,
      80,    81,    82,    83,    84,    19,   105,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     7,     0,    60,     0,     0,    20,     0,
       8,     0,     0,    68,     0,     0,   151,   150,   152,     0,
       0,     0,     0,     0,     0,     0,    38,    39,     0,     0,
       0,     0,     0,     0,   146,   147,   148,   149,     0,     0,
      45,     0,    23,     0,     0,    30,    33,    34,    35,    40,
       0,     0,     0,     0,     0,     0,     0,   137,   136,   135,
     116,   133,    27,   154,   153,     0,   134,    29,   193,   194,
     192,     5,     0,     0,    12,    14,    13,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      11,    18,    15,    16,     6,     9,     0,     0,     0,     0,
       0,    47,    46,   136,   135,   116,   134,   192,     0,     0,
       0,     0,   196,   196,     0,     0,     0,     0,     0,   134,
       0,   131,    36,    37,    32,    24,    48,     0,    50,    25,
     175,   179,   176,   178,   177,   140,   141,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    31,    26,    28,   196,     0,
       0,     0,     0,    77,    90,    99,    93,     0,   101,    97,
     110,   111,   106,   108,   107,   109,    75,    76,     0,     0,
       0,     0,     0,   168,     0,     0,     0,     0,     0,     0,
     200,   199,   134,     0,     0,     0,     0,     0,     0,     0,
     132,     0,     0,     0,     0,     0,     0,   123,   124,   125,
     126,   129,   130,   120,   127,   128,   117,   118,   119,   121,
     122,     0,   174,   134,   113,   115,     0,   187,   188,   189,
     190,   191,   112,   171,     0,   173,   205,   114,   196,   196,
      94,     0,   196,    98,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   196,   196,     0,   143,   144,
       0,     0,     0,     0,   180,     0,    49,     0,    52,     0,
      51,   139,   196,   142,   196,     0,     0,     0,     0,   100,
       0,    17,   136,   134,   192,     0,   136,   134,   192,     0,
     169,   156,     0,     0,   196,     0,   197,   198,     0,     0,
       0,     0,    55,    53,    54,     0,     0,   207,     0,    89,
       0,    92,     0,    96,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   142,   186,   206,     0,     0,     0,   163,   159,
     116,   192,     0,   162,   160,   116,     0,   170,   157,     0,
     138,   116,    41,   116,   116,     0,    88,    91,    95,     0,
       0,     0,     0,     0,     0,   155,     0,     0,     0,   133,
       0,   134,     0,     0,   205,     0,     0,     0,     0,     0,
       0,     0,   172,     0,     0,   158,     0,   161,     0,   166,
      42,    43,   182,   183,     0,     0,   164,     0,   167,   165,
     202,     0,   201,     0,   203,     0,     0,   204
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -408,  -408,  -408,  -408,   106,   -30,  -408,   393,  -408,  -408,
    -324,  -408,  -408,    -9,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,    -1,   -66,   -60,  -408,  -408,   142,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,  -408,     8,  -408,   -55,  -230,   263,
    -408,   -28,   -53,   -40,  -364,  -408,  -408,  -407,  -179,  -408,
    -408,   321,   200,  -254,    79,  -408,   -54,  -150,  -408,  -408,
    -408,  -408,     0
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
       0,     1,     8,     9,    50,   153,    51,    87,    17,    70,
     122,   123,   124,   125,   126,   127,   128,    19,   129,   130,
     131,   154,    53,   155,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,   156,    66,   270,   312,   197,
     137,   183,   184,   185,   141,   142,   143,   440,   264,   144,
     145,   232,   198,   450,   313,   314,   186,   273,   147,   482,
     148,   149,   187
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      18,   265,   483,    11,   304,   370,    14,    15,    16,    85,
     133,    52,    52,   164,   166,   135,   146,   139,   443,    52,
      65,    65,  -195,    96,    97,    98,   241,   233,    65,   242,
     140,    21,    22,    23,    24,    25,    26,    85,   389,   390,
     215,   216,   138,   274,   157,   158,   159,   160,   161,   223,
      10,   345,   346,   347,   348,   349,   350,     2,   210,   211,
     212,   355,   477,   213,   199,   214,   356,    52,    52,   132,
     150,   449,   152,     3,    86,    12,    65,    65,   134,   427,
      13,   168,   169,    52,    10,   281,   171,   172,   306,   173,
     105,    20,    65,   282,   188,   109,   110,   449,  -205,   449,
     238,   396,   441,    10,   114,   115,   116,   356,   117,  -195,
     319,   260,   322,   320,   233,   323,   283,   234,   119,    67,
      10,   231,   181,   285,   284,  -166,    71,   121,   238,   439,
      68,   286,   206,   207,   208,   398,   337,   238,   272,   272,
      16,     4,     5,     6,     7,   239,    69,    96,    97,    98,
      72,   238,   372,   279,  -145,  -145,  -145,  -145,  -145,    73,
     268,  -145,  -207,  -145,   138,   138,   355,    74,   357,   358,
      75,   356,   360,    76,    91,    92,   259,   261,   262,   303,
     215,   216,  -195,   395,   272,   376,   377,   233,    77,   223,
     170,   269,   150,   150,   275,   276,   277,    78,  -195,   399,
     391,   392,   385,   233,   386,  -145,   238,   472,   463,   473,
     138,   393,   394,    79,   105,   238,   400,   321,    80,   109,
     110,   165,   167,    81,   406,   199,    82,    10,   114,   115,
     116,    84,   117,    83,    88,    90,  -145,  -145,   150,  -145,
    -145,   316,   119,   442,   151,   174,  -145,   175,   228,   229,
     230,   121,   307,   176,   308,   309,   310,   311,   324,   177,
     326,   231,   178,   179,   272,   272,   180,   332,   272,   133,
     189,   363,   190,   367,   135,   146,   139,   191,   192,   202,
     193,   272,   272,   194,   195,   474,   196,   243,   244,   140,
     138,   138,   203,   204,   138,   205,   209,   362,   272,   366,
     272,   138,   235,   236,   485,   237,    85,   138,   138,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   150,   150,
     272,   371,   150,   254,   138,   364,   138,   368,   132,   150,
     255,   256,   257,   136,   375,   150,   150,   134,   241,   258,
     238,   379,   380,   133,   266,   267,   138,   318,   135,   146,
     139,   278,   150,   325,   150,   387,   388,   420,   327,   328,
     415,   425,   416,   140,   417,   182,   329,   431,   330,   433,
     434,   331,   333,   334,   150,   138,   335,   338,   339,   340,
     341,   451,   200,   342,   201,   352,   418,   419,   343,   353,
     423,   424,   354,   359,   361,   215,   216,   421,   365,   432,
     373,   421,   132,   150,   223,   374,   381,   451,   382,   451,
     383,   134,   384,   403,   414,   397,   404,   401,   446,   407,
     445,   409,   410,   412,   413,   422,   430,   436,   428,   435,
     437,   438,   447,   452,   448,   453,   462,   456,   459,   455,
     464,   454,   263,   263,   465,   460,   458,   467,   481,   461,
     470,   471,   469,   478,   439,   271,   271,   486,    89,   476,
     411,   226,   227,   228,   229,   230,   479,   240,   480,   215,
     216,     0,   484,     0,   219,   220,   231,   487,   223,   344,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,     0,   305,   402,     0,
       0,   271,   305,   315,     0,   317,     0,     0,   215,   216,
       0,   217,   218,   219,   220,   221,   222,   223,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    96,
      97,    98,     0,   224,   225,   226,   227,   228,   229,   230,
       0,     0,     0,     0,   305,   305,   305,   305,   305,   305,
     231,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   307,     0,   308,   309,   310,
     311,     0,   224,   225,   226,   227,   228,   229,   230,     0,
       0,   271,   271,     0,     0,   271,     0,     0,     0,   231,
       0,   369,   136,     0,   263,     0,   105,     0,   271,   271,
     305,   109,   110,   378,     0,     0,     0,     0,     0,    10,
     114,   115,   116,     0,   117,   271,     0,   271,     0,     0,
       0,     0,     0,     0,   119,     0,     0,     0,   305,     0,
       0,     0,     0,   121,     0,   444,   405,   271,     0,     0,
       0,     0,     0,     0,   215,   216,     0,   217,   218,   219,
     220,   221,   222,   223,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   426,   136,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   210,   211,   212,     0,   263,   213,     0,   214,   215,
     216,     0,   217,   218,   219,   220,   221,   222,   223,     0,
       0,     0,     0,     0,     0,     0,     0,   457,   224,   225,
     226,   227,   228,   229,   230,     0,     0,     0,   466,     0,
       0,     0,     0,     0,     0,   231,     0,   263,     0,     0,
       0,    21,    22,    23,    24,    25,    26,    93,    94,    95,
      96,    97,    98,    99,     0,   100,   263,   101,   102,   103,
       0,   104,     0,   224,   225,   226,   227,   228,   229,   230,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     231,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,     0,     0,   105,   106,   107,
     108,     0,   109,   110,     0,     0,   111,   112,   113,     0,
      10,   114,   115,   116,     0,   117,     0,     0,   118,     0,
       0,     0,     0,     0,     0,   119,     0,     0,     0,     0,
       0,    16,   120,     0,   121,    21,    22,    23,    24,    25,
      26,    93,    94,    95,    96,    97,    98,    99,     0,   100,
       0,   101,   102,   103,     0,   104,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,     0,
       0,   105,   106,   107,   108,     0,   109,   110,     0,     0,
     111,   112,   113,     0,    10,   114,   115,   116,     0,   117,
       0,     0,   118,     0,     0,     0,     0,     0,     0,   119,
       0,     0,     0,     0,     0,    16,   215,   216,   121,   217,
     218,   219,   220,   221,   222,   223,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   215,   216,     0,   217,   218,
     219,   220,   221,   222,   223,   215,   216,     0,   217,   218,
     219,   220,   221,   222,   223,   215,   216,     0,   217,   218,
     219,   220,   221,   222,   223,   215,   216,     0,   217,   218,
     219,   220,   221,   222,   223,     0,     0,     0,     0,     0,
     224,   225,   226,   227,   228,   229,   230,   215,   216,     0,
     217,   218,   219,   220,   221,   222,   223,   231,   351,   224,
     225,   226,   227,   228,   229,   230,     0,   280,     0,   224,
     225,   226,   227,   228,   229,   230,   231,     0,   336,   224,
     225,   226,   227,   228,   229,   230,   231,   408,     0,   224,
     225,   226,   227,   228,   229,   230,   231,   429,     0,     0,
       0,     0,     0,     0,     0,     0,   231,     0,     0,     0,
       0,   224,   225,   226,   227,   228,   229,   230,     0,   468,
       0,     0,     0,     0,     0,     0,   215,   216,   231,   217,
     218,   219,   220,   221,   222,   223,   215,   216,     0,   217,
     218,   219,   220,   221,   222,   223,   215,   216,     0,   217,
       0,   219,   220,   221,   222,   223,   215,   216,     0,     0,
       0,   219,   220,   221,   222,   223,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     224,   225,   226,   227,   228,   229,   230,     0,   475,     0,
     224,   225,   226,   227,   228,   229,   230,   231,     0,     0,
     224,   225,   226,   227,   228,   229,   230,   231,     0,     0,
     224,   225,   226,   227,   228,   229,   230,   231,    21,    22,
      23,    24,    25,    26,    93,     0,     0,   231,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,    23,    24,    25,    26,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,    23,    24,    25,    26,     0,     0,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,   162,   163,    32,    33,     0,
       0,     0,    37,    38,    39,    40,    41,     0,     0,    44,
       0,     0,    47,    48,    49,    21,    22,    23,    24,    25,
      26,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    32,    33,     0,     0,     0,    37,    38,    39,    40,
      41,     0,     0,    44,     0,     0,    47,    48,    49
};

static const yytype_int16 yycheck[] =
{
       9,   180,   107,     3,   234,   329,     6,     7,   113,    45,
      70,    12,    13,    79,    80,    70,    70,    70,   425,    20,
      12,    13,   107,    12,    13,    14,   112,   112,    20,   115,
      70,     3,     4,     5,     6,     7,     8,    45,   108,   109,
      36,    37,    70,   193,    74,    75,    76,    77,    78,    45,
      92,   281,   282,   283,   284,   285,   286,     0,    28,    29,
      30,   112,   469,    33,   118,    35,   117,    68,    69,    70,
      70,   435,    72,    16,   110,   107,    68,    69,    70,   403,
     107,    81,    82,    84,    92,   107,    86,    87,   238,    89,
      79,   107,    84,   115,   103,    84,    85,   461,   108,   463,
     107,   108,   112,    92,    93,    94,    95,   117,    97,   107,
     107,   177,   107,   110,   112,   110,   107,   115,   107,    13,
      92,   117,   111,   107,   115,   108,    20,   116,   107,   112,
     107,   115,   132,   133,   134,   365,   115,   107,   192,   193,
     113,    84,    85,    86,    87,   115,   107,    12,    13,    14,
     100,   107,   331,   109,    26,    27,    28,    29,    30,   100,
     190,    33,   108,    35,   192,   193,   112,   107,   318,   319,
     100,   117,   322,   100,    68,    69,   176,   177,   178,   233,
      36,    37,   107,   108,   238,   335,   336,   112,   100,    45,
      84,   191,   192,   193,   194,   195,   196,   100,   107,   108,
     108,   109,   352,   112,   354,    77,   107,   461,   109,   463,
     238,   108,   109,   100,    79,   107,   108,   247,   100,    84,
      85,    79,    80,   100,   374,   279,   100,    92,    93,    94,
      95,   109,    97,   108,   110,   108,   108,   109,   238,   111,
     112,   241,   107,   422,   108,   108,   118,   108,   104,   105,
     106,   116,    48,   107,    50,    51,    52,    53,   258,   107,
     260,   117,   107,   107,   318,   319,   107,   267,   322,   329,
      24,   325,   100,   327,   329,   329,   329,   107,   107,   111,
     107,   335,   336,   107,   107,   464,   107,   101,   101,   329,
     318,   319,   111,   111,   322,   111,   111,   325,   352,   327,
     354,   329,   111,   111,   483,   111,    45,   335,   336,   108,
     101,   109,   101,   101,   101,   101,   101,   101,   318,   319,
     374,   330,   322,   101,   352,   325,   354,   327,   329,   329,
     101,   101,   101,    70,   334,   335,   336,   329,   112,   107,
     107,   341,   342,   403,    19,   107,   374,   107,   403,   403,
     403,   101,   352,    22,   354,   355,   356,   397,    22,    22,
     390,   401,   392,   403,   394,   102,   108,   407,   108,   409,
     410,   107,   101,    22,   374,   403,   109,   108,   108,    99,
      22,   435,   119,    22,   121,   107,   395,   396,   115,   108,
     399,   400,   107,   101,   108,    36,    37,   397,   115,   408,
      99,   401,   403,   403,    45,   107,   100,   461,   108,   463,
     108,   403,   108,    18,   118,   112,   108,   112,    77,   112,
     429,   112,   112,   108,   108,   111,   108,   108,   111,   109,
     108,   108,   108,    25,   108,   108,   101,   108,    99,   111,
     107,   441,   179,   180,   453,    99,   446,   456,    83,   109,
     459,   460,   108,   108,   112,   192,   193,   108,    65,   468,
     381,   102,   103,   104,   105,   106,   475,   146,   477,    36,
      37,    -1,   481,    -1,    41,    42,   117,   486,    45,   279,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,    -1,   234,    26,    -1,
      -1,   238,   239,   240,    -1,   242,    -1,    -1,    36,    37,
      -1,    39,    40,    41,    42,    43,    44,    45,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    12,
      13,    14,    -1,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,    -1,    -1,   281,   282,   283,   284,   285,   286,
     117,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    -1,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   318,   319,    -1,    -1,   322,    -1,    -1,    -1,   117,
      -1,   328,   329,    -1,   331,    -1,    79,    -1,   335,   336,
     337,    84,    85,   340,    -1,    -1,    -1,    -1,    -1,    92,
      93,    94,    95,    -1,    97,   352,    -1,   354,    -1,    -1,
      -1,    -1,    -1,    -1,   107,    -1,    -1,    -1,   365,    -1,
      -1,    -1,    -1,   116,    -1,    27,   373,   374,    -1,    -1,
      -1,    -1,    -1,    -1,    36,    37,    -1,    39,    40,    41,
      42,    43,    44,    45,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   402,   403,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    28,    29,    30,    -1,   422,    33,    -1,    35,    36,
      37,    -1,    39,    40,    41,    42,    43,    44,    45,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   444,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,    -1,   455,    -1,
      -1,    -1,    -1,    -1,    -1,   117,    -1,   464,    -1,    -1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    -1,    17,   483,    19,    20,    21,
      -1,    23,    -1,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     117,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    79,    80,    81,
      82,    -1,    84,    85,    -1,    -1,    88,    89,    90,    -1,
      92,    93,    94,    95,    -1,    97,    -1,    -1,   100,    -1,
      -1,    -1,    -1,    -1,    -1,   107,    -1,    -1,    -1,    -1,
      -1,   113,   114,    -1,   116,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    -1,    17,
      -1,    19,    20,    21,    -1,    23,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    79,    80,    81,    82,    -1,    84,    85,    -1,    -1,
      88,    89,    90,    -1,    92,    93,    94,    95,    -1,    97,
      -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,    -1,   107,
      -1,    -1,    -1,    -1,    -1,   113,    36,    37,   116,    39,
      40,    41,    42,    43,    44,    45,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    36,    37,    -1,    39,    40,
      41,    42,    43,    44,    45,    36,    37,    -1,    39,    40,
      41,    42,    43,    44,    45,    36,    37,    -1,    39,    40,
      41,    42,    43,    44,    45,    36,    37,    -1,    39,    40,
      41,    42,    43,    44,    45,    -1,    -1,    -1,    -1,    -1,
     100,   101,   102,   103,   104,   105,   106,    36,    37,    -1,
      39,    40,    41,    42,    43,    44,    45,   117,   118,   100,
     101,   102,   103,   104,   105,   106,    -1,   108,    -1,   100,
     101,   102,   103,   104,   105,   106,   117,    -1,   109,   100,
     101,   102,   103,   104,   105,   106,   117,   108,    -1,   100,
     101,   102,   103,   104,   105,   106,   117,   108,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   117,    -1,    -1,    -1,
      -1,   100,   101,   102,   103,   104,   105,   106,    -1,   108,
      -1,    -1,    -1,    -1,    -1,    -1,    36,    37,   117,    39,
      40,    41,    42,    43,    44,    45,    36,    37,    -1,    39,
      40,    41,    42,    43,    44,    45,    36,    37,    -1,    39,
      -1,    41,    42,    43,    44,    45,    36,    37,    -1,    -1,
      -1,    41,    42,    43,    44,    45,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     100,   101,   102,   103,   104,   105,   106,    -1,   108,    -1,
     100,   101,   102,   103,   104,   105,   106,   117,    -1,    -1,
     100,   101,   102,   103,   104,   105,   106,   117,    -1,    -1,
     100,   101,   102,   103,   104,   105,   106,   117,     3,     4,
       5,     6,     7,     8,     9,    -1,    -1,   117,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    -1,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    57,    58,    59,    60,    -1,
      -1,    -1,    64,    65,    66,    67,    68,    -1,    -1,    71,
      -1,    -1,    74,    75,    76,     3,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    59,    60,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    -1,    -1,    71,    -1,    -1,    74,    75,    76
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,   120,     0,    16,    84,    85,    86,    87,   121,   122,
      92,   181,   107,   107,   181,   181,   113,   127,   132,   136,
     107,     3,     4,     5,     6,     7,     8,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
     123,   125,   140,   141,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   123,   107,   107,
     128,   123,   100,   100,   107,   100,   100,   100,   100,   100,
     100,   100,   100,   108,   109,    45,   110,   126,   110,   126,
     108,   123,   123,     9,    10,    11,    12,    13,    14,    15,
      17,    19,    20,    21,    23,    79,    80,    81,    82,    84,
      85,    88,    89,    90,    93,    94,    95,    97,   100,   107,
     114,   116,   129,   130,   131,   132,   133,   134,   135,   137,
     138,   139,   140,   142,   154,   156,   158,   159,   160,   161,
     162,   163,   164,   165,   168,   169,   175,   177,   179,   180,
     181,   108,   181,   124,   140,   142,   154,   124,   124,   124,
     124,   124,    57,    58,   141,   145,   141,   145,   181,   181,
     123,   181,   181,   181,   108,   108,   107,   107,   107,   107,
     107,   111,   158,   160,   161,   162,   175,   181,   132,    24,
     100,   107,   107,   107,   107,   107,   107,   158,   171,   175,
     158,   158,   111,   111,   111,   111,   181,   181,   181,   111,
      28,    29,    30,    33,    35,    36,    37,    39,    40,    41,
      42,    43,    44,    45,   100,   101,   102,   103,   104,   105,
     106,   117,   170,   112,   115,   111,   111,   111,   107,   115,
     170,   112,   115,   101,   101,   108,   101,   109,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   107,   181,
     141,   181,   181,   158,   167,   167,    19,   107,   124,   181,
     156,   158,   175,   176,   176,   181,   181,   181,   101,   109,
     108,   107,   115,   107,   115,   107,   115,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   175,   157,   158,   176,    48,    50,    51,
      52,    53,   157,   173,   174,   158,   181,   158,   107,   107,
     110,   124,   107,   110,   181,    22,   181,    22,    22,   108,
     108,   107,   181,   101,    22,   109,   109,   115,   108,   108,
      99,    22,    22,   115,   171,   157,   157,   157,   157,   157,
     157,   118,   107,   108,   107,   112,   117,   176,   176,   101,
     176,   108,   160,   175,   181,   115,   160,   175,   181,   158,
     129,   132,   167,    99,   107,   181,   176,   176,   158,   181,
     181,   100,   108,   108,   108,   176,   176,   181,   181,   108,
     109,   108,   109,   108,   109,   108,   108,   112,   157,   108,
     108,   112,    26,    18,   108,   158,   176,   112,   108,   112,
     112,   173,   108,   108,   118,   124,   124,   124,   132,   132,
     162,   181,   111,   132,   132,   162,   158,   129,   111,   108,
     108,   162,   132,   162,   162,   109,   108,   108,   108,   112,
     166,   112,   167,   166,    27,   132,    77,   108,   108,   163,
     172,   175,    25,   108,   181,   111,   108,   158,   181,    99,
      99,   109,   101,   109,   107,   132,   158,   132,   108,   108,
     132,   132,   172,   172,   167,   108,   132,   166,   108,   132,
     132,    83,   178,   107,   132,   167,   108,   132
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_uint8 yyr1[] =
{
       0,   119,   120,   120,   121,   122,   122,   122,   122,   122,
     123,   123,   124,   124,   124,   125,   125,   125,   125,   126,
     126,   127,   128,   128,   129,   129,   129,   129,   129,   129,
     129,   129,   129,   129,   129,   129,   129,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   138,   139,   139,
     139,   139,   139,   139,   139,   139,   140,   140,   140,   140,
     140,   140,   141,   141,   141,   141,   141,   141,   142,   143,
     143,   143,   143,   144,   145,   145,   145,   145,   145,   145,
     145,   145,   145,   145,   145,   146,   146,   146,   147,   147,
     147,   148,   148,   148,   148,   149,   149,   149,   149,   150,
     151,   152,   153,   154,   154,   154,   155,   155,   155,   155,
     155,   155,   156,   156,   156,   157,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   159,   160,
     161,   161,   162,   162,   162,   162,   163,   163,   163,   163,
     163,   163,   163,   164,   164,   165,   165,   165,   165,   165,
     165,   165,   165,   165,   165,   165,   166,   166,   167,   168,
     168,   169,   169,   169,   169,   170,   170,   170,   170,   170,
     171,   171,   172,   172,   172,   172,   173,   174,   174,   174,
     174,   174,   175,   175,   175,   175,   176,   176,   176,   176,
     176,   177,   177,   178,   178,   179,   179,   180,   181
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     0,     2,     2,     5,     5,     4,     4,     5,
       1,     3,     1,     1,     1,     3,     3,     6,     3,     0,
       1,     1,     0,     2,     2,     2,     2,     1,     2,     1,
       1,     2,     2,     1,     1,     1,     2,     2,     1,     1,
       3,     7,    10,    10,     1,     1,     2,     2,     2,     4,
       2,     4,     4,     5,     5,     5,     1,     1,     1,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     4,     4,     4,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     9,     7,
       4,     9,     7,     4,     5,     9,     7,     4,     5,     4,
       6,     4,     1,     1,     1,     1,     4,     4,     4,     4,
       4,     4,     3,     3,     3,     1,     1,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     2,     3,     1,     1,     1,     1,     1,     7,     4,
       2,     2,     4,     4,     4,     6,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     8,     5,     7,    10,     7,
       7,    10,     7,     7,    11,    12,     0,     5,     1,     5,
       7,     3,     9,     3,     3,     1,     1,     1,     1,     1,
       3,     1,     3,     3,     1,     1,     4,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     0,     3,     3,     1,
       1,    13,    12,     2,     5,     3,     6,     5,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 3: /* program: program function_def  */
#line 110 "lrparser.y"
                               {/* printf("LIST SIZE %d",frontEndContext.getFuncList().size())  ;*/ }
#line 2048 "y.tab.c"
    break;

  case 4: /* function_def: function_data function_body  */
#line 112 "lrparser.y"
                                            { 
	                                          Function* func=(Function*)(yyvsp[-1].node);
                                              blockStatement* block=(blockStatement*)(yyvsp[0].node);
                                              func->setBlockStatement(block);
											   Util::addFuncToList(func);
											}
#line 2059 "y.tab.c"
    break;

  case 5: /* function_data: T_FUNC id '(' paramList ')'  */
#line 119 "lrparser.y"
                                           { 
										   (yyval.node)=Util::createFuncNode((yyvsp[-3].node),(yyvsp[-1].pList)->PList);
                                           Util::setCurrentFuncType(GEN_FUNC);
										   Util::resetTemp(tempIds);
										   tempIds.clear();
	                                      }
#line 2070 "y.tab.c"
    break;

  case 6: /* function_data: T_STATIC id '(' paramList ')'  */
#line 125 "lrparser.y"
                                                           {
										   (yyval.node)=Util::createStaticFuncNode((yyvsp[-3].node),(yyvsp[-1].pList)->PList);
                                            Util::setCurrentFuncType(STATIC_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
	                                      }
#line 2081 "y.tab.c"
    break;

  case 7: /* function_data: T_INCREMENTAL '(' paramList ')'  */
#line 131 "lrparser.y"
                                                     { 
										   (yyval.node)=Util::createIncrementalNode((yyvsp[-1].pList)->PList);
                                            Util::setCurrentFuncType(INCREMENTAL_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
	                                      }
#line 2092 "y.tab.c"
    break;

  case 8: /* function_data: T_DECREMENTAL '(' paramList ')'  */
#line 137 "lrparser.y"
                                                             { 
										   (yyval.node)=Util::createDecrementalNode((yyvsp[-1].pList)->PList);
                                            Util::setCurrentFuncType(DECREMENTAL_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
	                                      }
#line 2103 "y.tab.c"
    break;

  case 9: /* function_data: T_DYNAMIC id '(' paramList ')'  */
#line 143 "lrparser.y"
                                                        { (yyval.node)=Util::createDynamicFuncNode((yyvsp[-3].node),(yyvsp[-1].pList)->PList);
                                            Util::setCurrentFuncType(DYNAMIC_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
											}
#line 2113 "y.tab.c"
    break;

  case 10: /* paramList: param  */
#line 150 "lrparser.y"
                 {(yyval.pList)=Util::createPList((yyvsp[0].node));}
#line 2119 "y.tab.c"
    break;

  case 11: /* paramList: param ',' paramList  */
#line 151 "lrparser.y"
                                     {(yyval.pList)=Util::addToPList((yyvsp[0].pList),(yyvsp[-2].node)); 
			                           }
#line 2126 "y.tab.c"
    break;

  case 12: /* type: type1  */
#line 154 "lrparser.y"
            {(yyval.node) = (yyvsp[0].node);}
#line 2132 "y.tab.c"
    break;

  case 13: /* type: type2  */
#line 155 "lrparser.y"
            {(yyval.node) = (yyvsp[0].node);}
#line 2138 "y.tab.c"
    break;

  case 14: /* type: type3  */
#line 156 "lrparser.y"
                {(yyval.node) = (yyvsp[0].node);}
#line 2144 "y.tab.c"
    break;

  case 15: /* param: type1 by_reference id  */
#line 158 "lrparser.y"
                              {  //Identifier* id=(Identifier*)Util::createIdentifierNode($3);
                        Type* type=(Type*)(yyvsp[-2].node);
	                     Identifier* id=(Identifier*)(yyvsp[0].node);
						 
						 if(type->isGraphType())
						    {
							 tempIds.push_back(id);
						   
							}
						if(type->isGNNType())
							{
							tempIds.push_back(id);
						
							}

					printf("\n");
                    (yyval.node)=Util::createParamNode((yyvsp[-2].node),(yyvsp[-1].bval),(yyvsp[0].node)); }
#line 2166 "y.tab.c"
    break;

  case 16: /* param: type2 by_reference id  */
#line 175 "lrparser.y"
                                       { // Identifier* id=(Identifier*)Util::createIdentifierNode($3);
			  
					
                             (yyval.node)=Util::createParamNode((yyvsp[-2].node),(yyvsp[-1].bval),(yyvsp[0].node));}
#line 2175 "y.tab.c"
    break;

  case 17: /* param: type2 by_reference id '(' id ')'  */
#line 179 "lrparser.y"
                                                              { // Identifier* id1=(Identifier*)Util::createIdentifierNode($5);
			                            //Identifier* id=(Identifier*)Util::createIdentifierNode($3);
				                        Type* tempType=(Type*)(yyvsp[-5].node);
			                            if(tempType->isNodeEdgeType())
										  tempType->addSourceGraph((yyvsp[-1].node));
				                         (yyval.node)=Util::createParamNode(tempType,(yyvsp[-4].bval),(yyvsp[-3].node));
									 }
#line 2187 "y.tab.c"
    break;

  case 18: /* param: type1 '&' id  */
#line 186 "lrparser.y"
                                               {
					Type* type = (Type*)(yyvsp[-2].node);
					type->setRefType();
					Identifier* id=(Identifier*)(yyvsp[0].node);
					if(type->isGraphType())
					{
						tempIds.push_back(id);
					}
					printf("\n");
                    (yyval.node)=Util::createParamNode((yyvsp[-2].node),(yyvsp[0].node));
				}
#line 2203 "y.tab.c"
    break;

  case 19: /* by_reference: %empty  */
#line 199 "lrparser.y"
                             {(yyval.bval) = false;}
#line 2209 "y.tab.c"
    break;

  case 20: /* by_reference: '&'  */
#line 200 "lrparser.y"
                      {(yyval.bval) = true;}
#line 2215 "y.tab.c"
    break;

  case 21: /* function_body: blockstatements  */
#line 206 "lrparser.y"
                                {(yyval.node)=(yyvsp[0].node);}
#line 2221 "y.tab.c"
    break;

  case 22: /* statements: %empty  */
#line 209 "lrparser.y"
              {}
#line 2227 "y.tab.c"
    break;

  case 23: /* statements: statements statement  */
#line 210 "lrparser.y"
                               {printf ("found one statement\n") ; Util::addToBlock((yyvsp[0].node)); }
#line 2233 "y.tab.c"
    break;

  case 24: /* statement: declaration ';'  */
#line 212 "lrparser.y"
                          {(yyval.node)=(yyvsp[-1].node);}
#line 2239 "y.tab.c"
    break;

  case 25: /* statement: assignment ';'  */
#line 213 "lrparser.y"
                       {printf ("found an assignment type statement" ); (yyval.node)=(yyvsp[-1].node);}
#line 2245 "y.tab.c"
    break;

  case 26: /* statement: proc_call ';'  */
#line 214 "lrparser.y"
                       {printf ("found an proc call type statement" );(yyval.node)=Util::createNodeForProcCallStmt((yyvsp[-1].node));}
#line 2251 "y.tab.c"
    break;

  case 27: /* statement: control_flow  */
#line 215 "lrparser.y"
                      {printf ("found an control flow type statement" );(yyval.node)=(yyvsp[0].node);}
#line 2257 "y.tab.c"
    break;

  case 28: /* statement: reduction ';'  */
#line 216 "lrparser.y"
                      {printf ("found an reduction type statement" );(yyval.node)=(yyvsp[-1].node);}
#line 2263 "y.tab.c"
    break;

  case 29: /* statement: bfs_abstraction  */
#line 217 "lrparser.y"
                          {printf ("found bfs\n") ;(yyval.node)=(yyvsp[0].node); }
#line 2269 "y.tab.c"
    break;

  case 30: /* statement: blockstatements  */
#line 218 "lrparser.y"
                          {printf ("found block\n") ;(yyval.node)=(yyvsp[0].node);}
#line 2275 "y.tab.c"
    break;

  case 31: /* statement: unary_expr ';'  */
#line 219 "lrparser.y"
                         {printf ("found unary\n") ;(yyval.node)=Util::createNodeForUnaryStatements((yyvsp[-1].node));}
#line 2281 "y.tab.c"
    break;

  case 32: /* statement: return_stmt ';'  */
#line 220 "lrparser.y"
                          {printf ("found return\n") ;(yyval.node) = (yyvsp[-1].node) ;}
#line 2287 "y.tab.c"
    break;

  case 33: /* statement: batch_blockstmt  */
#line 221 "lrparser.y"
                           {printf ("found batch\n") ;(yyval.node) = (yyvsp[0].node);}
#line 2293 "y.tab.c"
    break;

  case 34: /* statement: on_add_blockstmt  */
#line 222 "lrparser.y"
                           {printf ("found on add block\n") ;(yyval.node) = (yyvsp[0].node);}
#line 2299 "y.tab.c"
    break;

  case 35: /* statement: on_delete_blockstmt  */
#line 223 "lrparser.y"
                              {printf ("found delete block\n") ;(yyval.node) = (yyvsp[0].node);}
#line 2305 "y.tab.c"
    break;

  case 36: /* statement: break_stmt ';'  */
#line 224 "lrparser.y"
                         {printf ("found break\n") ;(yyval.node) = Util::createNodeForBreakStatement();}
#line 2311 "y.tab.c"
    break;

  case 37: /* statement: continue_stmt ';'  */
#line 225 "lrparser.y"
                            {printf ("found continue\n") ;(yyval.node) = Util::createNodeForContinueStatement();}
#line 2317 "y.tab.c"
    break;

  case 38: /* break_stmt: T_BREAK  */
#line 227 "lrparser.y"
                     {printf ("found break\n") ;}
#line 2323 "y.tab.c"
    break;

  case 39: /* continue_stmt: T_CONTINUE  */
#line 228 "lrparser.y"
                           {printf ("found continue\n") ;}
#line 2329 "y.tab.c"
    break;

  case 40: /* blockstatements: block_begin statements block_end  */
#line 230 "lrparser.y"
                                                   { (yyval.node)=Util::finishBlock();}
#line 2335 "y.tab.c"
    break;

  case 41: /* batch_blockstmt: T_BATCH '(' id ':' expression ')' blockstatements  */
#line 232 "lrparser.y"
                                                                    {(yyval.node) = Util::createBatchBlockStmt((yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node));}
#line 2341 "y.tab.c"
    break;

  case 42: /* on_add_blockstmt: T_ONADD '(' id T_IN id '.' proc_call ')' ':' blockstatements  */
#line 234 "lrparser.y"
                                                                                {(yyval.node) = Util::createOnAddBlock((yyvsp[-7].node), (yyvsp[-5].node), (yyvsp[-3].node), (yyvsp[0].node));}
#line 2347 "y.tab.c"
    break;

  case 43: /* on_delete_blockstmt: T_ONDELETE '(' id T_IN id '.' proc_call ')' ':' blockstatements  */
#line 236 "lrparser.y"
                                                                                      {(yyval.node) = Util::createOnDeleteBlock((yyvsp[-7].node), (yyvsp[-5].node), (yyvsp[-3].node), (yyvsp[0].node));}
#line 2353 "y.tab.c"
    break;

  case 44: /* block_begin: '{'  */
#line 238 "lrparser.y"
                { Util::createNewBlock(); }
#line 2359 "y.tab.c"
    break;

  case 46: /* return_stmt: T_RETURN expression  */
#line 242 "lrparser.y"
                                  {(yyval.node) = Util::createReturnStatementNode((yyvsp[0].node));}
#line 2365 "y.tab.c"
    break;

  case 47: /* return_stmt: T_RETURN ';'  */
#line 243 "lrparser.y"
                       {(yyval.node) = Util::createReturnStatementNode(NULL);}
#line 2371 "y.tab.c"
    break;

  case 48: /* declaration: type1 id  */
#line 246 "lrparser.y"
                         {
	                     Type* type=(Type*)(yyvsp[-1].node);
	                     Identifier* id=(Identifier*)(yyvsp[0].node);
						 
						 if(type->isGraphType())
						    Util::storeGraphId(id);

                         (yyval.node)=Util::createNormalDeclNode((yyvsp[-1].node),(yyvsp[0].node));}
#line 2384 "y.tab.c"
    break;

  case 49: /* declaration: type1 id '=' rhs  */
#line 254 "lrparser.y"
                            {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                    
	                    (yyval.node)=Util::createAssignedDeclNode((yyvsp[-3].node),(yyvsp[-2].node),(yyvsp[0].node));}
#line 2392 "y.tab.c"
    break;

  case 50: /* declaration: type2 id  */
#line 257 "lrparser.y"
                    {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	            
                         (yyval.node)=Util::createNormalDeclNode((yyvsp[-1].node),(yyvsp[0].node)); }
#line 2400 "y.tab.c"
    break;

  case 51: /* declaration: type2 id '=' rhs  */
#line 260 "lrparser.y"
                           {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    (yyval.node)=Util::createAssignedDeclNode((yyvsp[-3].node),(yyvsp[-2].node),(yyvsp[0].node));}
#line 2408 "y.tab.c"
    break;

  case 52: /* declaration: type3 id '=' rhs  */
#line 263 "lrparser.y"
                           {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    (yyval.node)=Util::createAssignedDeclNode((yyvsp[-3].node),(yyvsp[-2].node),(yyvsp[0].node));}
#line 2416 "y.tab.c"
    break;

  case 53: /* declaration: type3 id '(' rhs ')'  */
#line 267 "lrparser.y"
                               {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    (yyval.node)=Util::createParamDeclNode((yyvsp[-4].node),(yyvsp[-3].node),(yyvsp[-1].node));}
#line 2424 "y.tab.c"
    break;

  case 54: /* declaration: type2 id '(' rhs ')'  */
#line 271 "lrparser.y"
                               {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    (yyval.node)=Util::createParamDeclNode((yyvsp[-4].node),(yyvsp[-3].node),(yyvsp[-1].node));}
#line 2432 "y.tab.c"
    break;

  case 55: /* declaration: type1 id '(' rhs ')'  */
#line 275 "lrparser.y"
                               {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    (yyval.node)=Util::createParamDeclNode((yyvsp[-4].node),(yyvsp[-3].node),(yyvsp[-1].node));}
#line 2440 "y.tab.c"
    break;

  case 56: /* type1: primitive  */
#line 280 "lrparser.y"
                 {(yyval.node)=(yyvsp[0].node); }
#line 2446 "y.tab.c"
    break;

  case 57: /* type1: graph  */
#line 281 "lrparser.y"
                {(yyval.node)=(yyvsp[0].node);}
#line 2452 "y.tab.c"
    break;

  case 58: /* type1: collections  */
#line 282 "lrparser.y"
                      { (yyval.node)=(yyvsp[0].node);}
#line 2458 "y.tab.c"
    break;

  case 59: /* type1: structs  */
#line 283 "lrparser.y"
                  {(yyval.node)=(yyvsp[0].node);}
#line 2464 "y.tab.c"
    break;

  case 60: /* type1: type1 T_ASTERISK  */
#line 284 "lrparser.y"
                           {
						Type* type=(Type*)(yyvsp[-1].node);
						type->incrementPointerStarCount();
						(yyval.node)=(yyvsp[-1].node);
						}
#line 2474 "y.tab.c"
    break;

  case 61: /* type1: gnn  */
#line 289 "lrparser.y"
              {(yyval.node)=(yyvsp[0].node);}
#line 2480 "y.tab.c"
    break;

  case 62: /* primitive: T_INT  */
#line 292 "lrparser.y"
                 { (yyval.node)=Util::createPrimitiveTypeNode(TYPE_INT);}
#line 2486 "y.tab.c"
    break;

  case 63: /* primitive: T_FLOAT  */
#line 293 "lrparser.y"
                  { (yyval.node)=Util::createPrimitiveTypeNode(TYPE_FLOAT);}
#line 2492 "y.tab.c"
    break;

  case 64: /* primitive: T_BOOL  */
#line 294 "lrparser.y"
                 { (yyval.node)=Util::createPrimitiveTypeNode(TYPE_BOOL);}
#line 2498 "y.tab.c"
    break;

  case 65: /* primitive: T_DOUBLE  */
#line 295 "lrparser.y"
                   { (yyval.node)=Util::createPrimitiveTypeNode(TYPE_DOUBLE); }
#line 2504 "y.tab.c"
    break;

  case 66: /* primitive: T_LONG  */
#line 296 "lrparser.y"
             {(yyval.node)=(yyval.node)=Util::createPrimitiveTypeNode(TYPE_LONG);}
#line 2510 "y.tab.c"
    break;

  case 67: /* primitive: T_STRING  */
#line 297 "lrparser.y"
                   {(yyval.node)=(yyval.node)=Util::createPrimitiveTypeNode(TYPE_STRING);}
#line 2516 "y.tab.c"
    break;

  case 68: /* type3: T_AUTOREF  */
#line 299 "lrparser.y"
                 { (yyval.node)=Util::createPrimitiveTypeNode(TYPE_AUTOREF);}
#line 2522 "y.tab.c"
    break;

  case 69: /* graph: T_GRAPH  */
#line 301 "lrparser.y"
                { (yyval.node)=Util::createGraphTypeNode(TYPE_GRAPH,NULL);}
#line 2528 "y.tab.c"
    break;

  case 70: /* graph: T_DIR_GRAPH  */
#line 302 "lrparser.y"
                     {(yyval.node)=Util::createGraphTypeNode(TYPE_DIRGRAPH,NULL);}
#line 2534 "y.tab.c"
    break;

  case 71: /* graph: T_GEOMCOMPLETEGRAPH  */
#line 303 "lrparser.y"
                              { (yyval.node)=Util::createGraphTypeNode(TYPE_GEOMCOMPLETEGRAPH,NULL);}
#line 2540 "y.tab.c"
    break;

  case 72: /* graph: T_GRAPH_LIST  */
#line 304 "lrparser.y"
                       { (yyval.node)=Util::createGraphTypeNode(TYPE_GRAPH_LIST,NULL);}
#line 2546 "y.tab.c"
    break;

  case 73: /* gnn: T_GNN  */
#line 307 "lrparser.y"
            { (yyval.node)=Util::createGNNTypeNode(TYPE_GNN,NULL);}
#line 2552 "y.tab.c"
    break;

  case 74: /* collections: T_LIST  */
#line 309 "lrparser.y"
                     { (yyval.node)=Util::createCollectionTypeNode(TYPE_LIST,NULL);}
#line 2558 "y.tab.c"
    break;

  case 75: /* collections: T_SET_NODES '<' id '>'  */
#line 310 "lrparser.y"
                                         {//Identifier* id=(Identifier*)Util::createIdentifierNode($3);
			                     (yyval.node)=Util::createCollectionTypeNode(TYPE_SETN,(yyvsp[-1].node));}
#line 2565 "y.tab.c"
    break;

  case 76: /* collections: T_SET_EDGES '<' id '>'  */
#line 312 "lrparser.y"
                                 {// Identifier* id=(Identifier*)Util::createIdentifierNode($3);
					                    (yyval.node)=Util::createCollectionTypeNode(TYPE_SETE,(yyvsp[-1].node));}
#line 2572 "y.tab.c"
    break;

  case 77: /* collections: T_UPDATES '<' id '>'  */
#line 314 "lrparser.y"
                                         { (yyval.node)=Util::createCollectionTypeNode(TYPE_UPDATES,(yyvsp[-1].node));}
#line 2578 "y.tab.c"
    break;

  case 78: /* collections: container  */
#line 315 "lrparser.y"
                        {(yyval.node) = (yyvsp[0].node);}
#line 2584 "y.tab.c"
    break;

  case 79: /* collections: vector  */
#line 316 "lrparser.y"
                 {(yyval.node) = (yyvsp[0].node);}
#line 2590 "y.tab.c"
    break;

  case 80: /* collections: set  */
#line 317 "lrparser.y"
                         {(yyval.node) = (yyvsp[0].node);}
#line 2596 "y.tab.c"
    break;

  case 81: /* collections: nodemap  */
#line 318 "lrparser.y"
                            {(yyval.node) = (yyvsp[0].node);}
#line 2602 "y.tab.c"
    break;

  case 82: /* collections: hashmap  */
#line 319 "lrparser.y"
                          {(yyval.node) = (yyvsp[0].node);}
#line 2608 "y.tab.c"
    break;

  case 83: /* collections: hashset  */
#line 320 "lrparser.y"
                      {(yyval.node) = (yyvsp[0].node);}
#line 2614 "y.tab.c"
    break;

  case 84: /* collections: btree  */
#line 321 "lrparser.y"
                        {(yyval.node) = (yyvsp[0].node);}
#line 2620 "y.tab.c"
    break;

  case 85: /* structs: T_POINT  */
#line 323 "lrparser.y"
                 { (yyval.node)=Util::createPointTypeNode(TYPE_POINT);}
#line 2626 "y.tab.c"
    break;

  case 86: /* structs: T_UNDIREDGE  */
#line 324 "lrparser.y"
                      { (yyval.node)=Util::createUndirectedEdgeTypeNode(TYPE_UNDIREDGE);}
#line 2632 "y.tab.c"
    break;

  case 87: /* structs: T_TRIANGLE  */
#line 325 "lrparser.y"
                     { (yyval.node)=Util::createTriangleTypeNode(TYPE_TRIANGLE);}
#line 2638 "y.tab.c"
    break;

  case 88: /* container: T_CONTAINER '<' type '>' '(' arg_list ',' type ')'  */
#line 327 "lrparser.y"
                                                               {(yyval.node) = Util::createContainerTypeNode(TYPE_CONTAINER, (yyvsp[-6].node), (yyvsp[-3].aList)->AList, (yyvsp[-1].node));}
#line 2644 "y.tab.c"
    break;

  case 89: /* container: T_CONTAINER '<' type '>' '(' arg_list ')'  */
#line 328 "lrparser.y"
                                                      { (yyval.node) =  Util::createContainerTypeNode(TYPE_CONTAINER, (yyvsp[-4].node), (yyvsp[-1].aList)->AList, NULL);}
#line 2650 "y.tab.c"
    break;

  case 90: /* container: T_CONTAINER '<' type '>'  */
#line 329 "lrparser.y"
                                     { list<argument*> argList;
			                          (yyval.node) = Util::createContainerTypeNode(TYPE_CONTAINER, (yyvsp[-1].node), argList, NULL);}
#line 2657 "y.tab.c"
    break;

  case 91: /* vector: T_VECTOR '<' type '>' '(' arg_list ',' type ')'  */
#line 332 "lrparser.y"
                                                       {(yyval.node) = Util::createContainerTypeNode(TYPE_VECTOR, (yyvsp[-6].node), (yyvsp[-3].aList)->AList, (yyvsp[-1].node));}
#line 2663 "y.tab.c"
    break;

  case 92: /* vector: T_VECTOR '<' type '>' '(' arg_list ')'  */
#line 333 "lrparser.y"
                                                  { (yyval.node) =  Util::createContainerTypeNode(TYPE_VECTOR, (yyvsp[-4].node), (yyvsp[-1].aList)->AList, NULL);}
#line 2669 "y.tab.c"
    break;

  case 93: /* vector: T_VECTOR '<' type '>'  */
#line 334 "lrparser.y"
                                 { list<argument*> argList;
			                          (yyval.node) = Util::createContainerTypeNode(TYPE_VECTOR, (yyvsp[-1].node), argList, NULL);}
#line 2676 "y.tab.c"
    break;

  case 94: /* vector: T_VECTOR '<' type '>' '&'  */
#line 336 "lrparser.y"
                                             { list<argument*> argList;
			                          (yyval.node) = Util::createContainerTypeRefNode(TYPE_VECTOR, (yyvsp[-2].node), argList, NULL);}
#line 2683 "y.tab.c"
    break;

  case 95: /* set: T_SET '<' type '>' '(' arg_list ',' type ')'  */
#line 339 "lrparser.y"
                                                 {(yyval.node) = Util::createContainerTypeNode(TYPE_SET, (yyvsp[-6].node), (yyvsp[-3].aList)->AList, (yyvsp[-1].node));}
#line 2689 "y.tab.c"
    break;

  case 96: /* set: T_SET '<' type '>' '(' arg_list ')'  */
#line 340 "lrparser.y"
                                               { (yyval.node) =  Util::createContainerTypeNode(TYPE_SET, (yyvsp[-4].node), (yyvsp[-1].aList)->AList, NULL);}
#line 2695 "y.tab.c"
    break;

  case 97: /* set: T_SET '<' type '>'  */
#line 341 "lrparser.y"
                              { list<argument*> argList;
			                          (yyval.node) = Util::createContainerTypeNode(TYPE_SET, (yyvsp[-1].node), argList, NULL);}
#line 2702 "y.tab.c"
    break;

  case 98: /* set: T_SET '<' type '>' '&'  */
#line 343 "lrparser.y"
                                           { list<argument*> argList;
			                          (yyval.node) = Util::createContainerTypeRefNode(TYPE_SET, (yyvsp[-2].node), argList, NULL);}
#line 2709 "y.tab.c"
    break;

  case 99: /* nodemap: T_NODEMAP '(' type ')'  */
#line 346 "lrparser.y"
                                 {(yyval.node) = Util::createNodeMapTypeNode(TYPE_NODEMAP, (yyvsp[-1].node));}
#line 2715 "y.tab.c"
    break;

  case 100: /* hashmap: T_HASHMAP '<' type ',' type '>'  */
#line 348 "lrparser.y"
                                          { list<argument*> argList;
			                          (yyval.node) = Util::createHashMapTypeNode(TYPE_HASHMAP, (yyvsp[-3].node), argList, (yyvsp[-1].node));}
#line 2722 "y.tab.c"
    break;

  case 101: /* hashset: T_HASHSET '<' type '>'  */
#line 351 "lrparser.y"
                                 { list<argument*> argList;
			                          (yyval.node) = Util::createHashSetTypeNode(TYPE_HASHSET, (yyvsp[-1].node), argList, NULL);}
#line 2729 "y.tab.c"
    break;

  case 102: /* btree: T_BTREE  */
#line 354 "lrparser.y"
                { (yyval.node) = Util::createBtreeTypeNode(TYPE_BTREE);}
#line 2735 "y.tab.c"
    break;

  case 103: /* type2: T_NODE  */
#line 356 "lrparser.y"
               {(yyval.node)=Util::createNodeEdgeTypeNode(TYPE_NODE) ;}
#line 2741 "y.tab.c"
    break;

  case 104: /* type2: T_EDGE  */
#line 357 "lrparser.y"
                {(yyval.node)=Util::createNodeEdgeTypeNode(TYPE_EDGE);}
#line 2747 "y.tab.c"
    break;

  case 105: /* type2: property  */
#line 358 "lrparser.y"
                      {(yyval.node)=(yyvsp[0].node);}
#line 2753 "y.tab.c"
    break;

  case 106: /* property: T_NP '<' primitive '>'  */
#line 360 "lrparser.y"
                                  { (yyval.node)=Util::createPropertyTypeNode(TYPE_PROPNODE,(yyvsp[-1].node)); }
#line 2759 "y.tab.c"
    break;

  case 107: /* property: T_EP '<' primitive '>'  */
#line 361 "lrparser.y"
                                       { (yyval.node)=Util::createPropertyTypeNode(TYPE_PROPEDGE,(yyvsp[-1].node)); }
#line 2765 "y.tab.c"
    break;

  case 108: /* property: T_NP '<' collections '>'  */
#line 362 "lrparser.y"
                                                    {  (yyval.node)=Util::createPropertyTypeNode(TYPE_PROPNODE,(yyvsp[-1].node)); }
#line 2771 "y.tab.c"
    break;

  case 109: /* property: T_EP '<' collections '>'  */
#line 363 "lrparser.y"
                                                     {(yyval.node)=Util::createPropertyTypeNode(TYPE_PROPEDGE,(yyvsp[-1].node));}
#line 2777 "y.tab.c"
    break;

  case 110: /* property: T_NP '<' T_NODE '>'  */
#line 364 "lrparser.y"
                                    {ASTNode* type = Util::createNodeEdgeTypeNode(TYPE_NODE);
			                         (yyval.node)=Util::createPropertyTypeNode(TYPE_PROPNODE, type); }
#line 2784 "y.tab.c"
    break;

  case 111: /* property: T_NP '<' T_EDGE '>'  */
#line 366 "lrparser.y"
                                                {ASTNode* type = Util::createNodeEdgeTypeNode(TYPE_EDGE);
			                         (yyval.node)=Util::createPropertyTypeNode(TYPE_PROPNODE, type); }
#line 2791 "y.tab.c"
    break;

  case 112: /* assignment: leftSide '=' rhs  */
#line 369 "lrparser.y"
                                { printf("testassign\n");(yyval.node)=Util::createAssignmentNode((yyvsp[-2].node),(yyvsp[0].node));}
#line 2797 "y.tab.c"
    break;

  case 113: /* assignment: indexExpr '=' rhs  */
#line 370 "lrparser.y"
                                  {printf ("called assign for count\n") ; (yyval.node)=Util::createAssignmentNode((yyvsp[-2].node) , (yyvsp[0].node));}
#line 2803 "y.tab.c"
    break;

  case 114: /* assignment: id '=' expression  */
#line 371 "lrparser.y"
                                               { (yyval.node) = Util::createAssignmentNode((yyvsp[-2].node), (yyvsp[0].node)); }
#line 2809 "y.tab.c"
    break;

  case 115: /* rhs: expression  */
#line 374 "lrparser.y"
                 { (yyval.node)=(yyvsp[0].node);}
#line 2815 "y.tab.c"
    break;

  case 116: /* expression: proc_call  */
#line 376 "lrparser.y"
                       { (yyval.node)=(yyvsp[0].node);}
#line 2821 "y.tab.c"
    break;

  case 117: /* expression: expression '+' expression  */
#line 377 "lrparser.y"
                                         { (yyval.node)=Util::createNodeForArithmeticExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_ADD);}
#line 2827 "y.tab.c"
    break;

  case 118: /* expression: expression '-' expression  */
#line 378 "lrparser.y"
                                             { (yyval.node)=Util::createNodeForArithmeticExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_SUB);}
#line 2833 "y.tab.c"
    break;

  case 119: /* expression: expression '*' expression  */
#line 379 "lrparser.y"
                                             {(yyval.node)=Util::createNodeForArithmeticExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_MUL);}
#line 2839 "y.tab.c"
    break;

  case 120: /* expression: expression T_ASTERISK expression  */
#line 380 "lrparser.y"
                                                            {(yyval.node)=Util::createNodeForArithmeticExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_MUL);}
#line 2845 "y.tab.c"
    break;

  case 121: /* expression: expression '/' expression  */
#line 381 "lrparser.y"
                                           {(yyval.node)=Util::createNodeForArithmeticExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_DIV);}
#line 2851 "y.tab.c"
    break;

  case 122: /* expression: expression '%' expression  */
#line 382 "lrparser.y"
                                                    {(yyval.node)=Util::createNodeForArithmeticExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_MOD);}
#line 2857 "y.tab.c"
    break;

  case 123: /* expression: expression T_AND_OP expression  */
#line 383 "lrparser.y"
                                              {(yyval.node)=Util::createNodeForLogicalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_AND);}
#line 2863 "y.tab.c"
    break;

  case 124: /* expression: expression T_OR_OP expression  */
#line 384 "lrparser.y"
                                                  {(yyval.node)=Util::createNodeForLogicalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_OR);}
#line 2869 "y.tab.c"
    break;

  case 125: /* expression: expression T_LE_OP expression  */
#line 385 "lrparser.y"
                                                 {(yyval.node)=Util::createNodeForRelationalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_LE);}
#line 2875 "y.tab.c"
    break;

  case 126: /* expression: expression T_GE_OP expression  */
#line 386 "lrparser.y"
                                                {(yyval.node)=Util::createNodeForRelationalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_GE);}
#line 2881 "y.tab.c"
    break;

  case 127: /* expression: expression '<' expression  */
#line 387 "lrparser.y"
                                                    {(yyval.node)=Util::createNodeForRelationalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_LT);}
#line 2887 "y.tab.c"
    break;

  case 128: /* expression: expression '>' expression  */
#line 388 "lrparser.y"
                                                    {(yyval.node)=Util::createNodeForRelationalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_GT);}
#line 2893 "y.tab.c"
    break;

  case 129: /* expression: expression T_EQ_OP expression  */
#line 389 "lrparser.y"
                                                        {(yyval.node)=Util::createNodeForRelationalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_EQ);}
#line 2899 "y.tab.c"
    break;

  case 130: /* expression: expression T_NE_OP expression  */
#line 390 "lrparser.y"
                                            {(yyval.node)=Util::createNodeForRelationalExpr((yyvsp[-2].node),(yyvsp[0].node),OPERATOR_NE);}
#line 2905 "y.tab.c"
    break;

  case 131: /* expression: '!' expression  */
#line 391 "lrparser.y"
                                         {(yyval.node)=Util::createNodeForUnaryExpr((yyvsp[0].node),OPERATOR_NOT);}
#line 2911 "y.tab.c"
    break;

  case 132: /* expression: '(' expression ')'  */
#line 392 "lrparser.y"
                                          { Expression* expr=(Expression*)(yyvsp[-1].node);
				                     expr->setEnclosedBrackets();
			                        (yyval.node)=expr;}
#line 2919 "y.tab.c"
    break;

  case 133: /* expression: val  */
#line 395 "lrparser.y"
                       {(yyval.node)=(yyvsp[0].node);}
#line 2925 "y.tab.c"
    break;

  case 134: /* expression: leftSide  */
#line 396 "lrparser.y"
                                    { (yyval.node)=Util::createNodeForId((yyvsp[0].node));}
#line 2931 "y.tab.c"
    break;

  case 135: /* expression: unary_expr  */
#line 397 "lrparser.y"
                                      {(yyval.node)=(yyvsp[0].node);}
#line 2937 "y.tab.c"
    break;

  case 136: /* expression: indexExpr  */
#line 398 "lrparser.y"
                                     {(yyval.node) = (yyvsp[0].node);}
#line 2943 "y.tab.c"
    break;

  case 137: /* expression: alloca_expr  */
#line 399 "lrparser.y"
                                       {(yyval.node)= (yyvsp[0].node);}
#line 2949 "y.tab.c"
    break;

  case 138: /* alloca_expr: T_ALLOCATE '<' type '>' '(' arg_list ')'  */
#line 401 "lrparser.y"
                                                       { 
					(yyval.node) = Util::createNodeForAllocaExpr((yyvsp[-4].node), (yyvsp[-1].aList)->AList); 
				}
#line 2957 "y.tab.c"
    break;

  case 139: /* indexExpr: expression '[' expression ']'  */
#line 405 "lrparser.y"
                                          {printf("first done this \n");(yyval.node) = Util::createNodeForIndexExpr((yyvsp[-3].node), (yyvsp[-1].node), OPERATOR_INDEX);}
#line 2963 "y.tab.c"
    break;

  case 140: /* unary_expr: expression T_INC_OP  */
#line 407 "lrparser.y"
                                   {(yyval.node)=Util::createNodeForUnaryExpr((yyvsp[-1].node),OPERATOR_INC);}
#line 2969 "y.tab.c"
    break;

  case 141: /* unary_expr: expression T_DEC_OP  */
#line 408 "lrparser.y"
                                                {(yyval.node)=Util::createNodeForUnaryExpr((yyvsp[-1].node),OPERATOR_DEC);}
#line 2975 "y.tab.c"
    break;

  case 142: /* proc_call: leftSide '(' arg_list ')'  */
#line 410 "lrparser.y"
                                      { 
										ASTNode* proc_callId = (yyvsp[-3].node);
										if(proc_callId->getTypeofNode()==NODE_ID){
											Identifier* id = (Identifier*)proc_callId;
											if(strcmp(id->getIdentifier(),"tsort")==0){
												frontEndContext.setThrustUsed(true);
											}
										}
                                       (yyval.node) = Util::createNodeForProcCall((yyvsp[-3].node),(yyvsp[-1].aList)->AList,NULL); 

									    }
#line 2991 "y.tab.c"
    break;

  case 143: /* proc_call: T_INCREMENTAL '(' arg_list ')'  */
#line 421 "lrparser.y"
                                                         { ASTNode* id = Util::createIdentifierNode("Incremental");
			                                   (yyval.node) = Util::createNodeForProcCall(id, (yyvsp[-1].aList)->AList,NULL); 

				                               }
#line 3000 "y.tab.c"
    break;

  case 144: /* proc_call: T_DECREMENTAL '(' arg_list ')'  */
#line 425 "lrparser.y"
                                                         { ASTNode* id = Util::createIdentifierNode("Decremental");
			                                   (yyval.node) = Util::createNodeForProcCall(id, (yyvsp[-1].aList)->AList,NULL); 

				                               }
#line 3009 "y.tab.c"
    break;

  case 145: /* proc_call: indexExpr '.' leftSide '(' arg_list ')'  */
#line 429 "lrparser.y"
                                                                  {
                                                   
													 Expression* expr = (Expression*)(yyvsp[-5].node);
                                                     (yyval.node) = Util::createNodeForProcCall((yyvsp[-3].node) , (yyvsp[-1].aList)->AList, expr); 

									                 }
#line 3020 "y.tab.c"
    break;

  case 146: /* val: INT_NUM  */
#line 440 "lrparser.y"
              { (yyval.node) = Util::createNodeForIval((yyvsp[0].ival)); }
#line 3026 "y.tab.c"
    break;

  case 147: /* val: FLOAT_NUM  */
#line 441 "lrparser.y"
                    {(yyval.node) = Util::createNodeForFval((yyvsp[0].fval));}
#line 3032 "y.tab.c"
    break;

  case 148: /* val: BOOL_VAL  */
#line 442 "lrparser.y"
                   { (yyval.node) = Util::createNodeForBval((yyvsp[0].bval));}
#line 3038 "y.tab.c"
    break;

  case 149: /* val: STRING_VAL  */
#line 443 "lrparser.y"
                     { (yyval.node) = Util::createNodeForSval((yyvsp[0].text));}
#line 3044 "y.tab.c"
    break;

  case 150: /* val: T_INF  */
#line 444 "lrparser.y"
                {(yyval.node)=Util::createNodeForINF(true);}
#line 3050 "y.tab.c"
    break;

  case 151: /* val: T_P_INF  */
#line 445 "lrparser.y"
                  {(yyval.node)=Util::createNodeForINF(true);}
#line 3056 "y.tab.c"
    break;

  case 152: /* val: T_N_INF  */
#line 446 "lrparser.y"
                  {(yyval.node)=Util::createNodeForINF(false);}
#line 3062 "y.tab.c"
    break;

  case 153: /* control_flow: selection_cf  */
#line 449 "lrparser.y"
                            { (yyval.node)=(yyvsp[0].node); }
#line 3068 "y.tab.c"
    break;

  case 154: /* control_flow: iteration_cf  */
#line 450 "lrparser.y"
                             { (yyval.node)=(yyvsp[0].node); }
#line 3074 "y.tab.c"
    break;

  case 155: /* iteration_cf: T_FIXEDPOINT T_UNTIL '(' id ':' expression ')' blockstatements  */
#line 452 "lrparser.y"
                                                                              { (yyval.node)=Util::createNodeForFixedPointStmt((yyvsp[-4].node),(yyvsp[-2].node),(yyvsp[0].node));}
#line 3080 "y.tab.c"
    break;

  case 156: /* iteration_cf: T_WHILE '(' boolean_expr ')' blockstatements  */
#line 453 "lrparser.y"
                                                                 {(yyval.node)=Util::createNodeForWhileStmt((yyvsp[-2].node),(yyvsp[0].node)); }
#line 3086 "y.tab.c"
    break;

  case 157: /* iteration_cf: T_DO blockstatements T_WHILE '(' boolean_expr ')' ';'  */
#line 454 "lrparser.y"
                                                                           {(yyval.node)=Util::createNodeForDoWhileStmt((yyvsp[-2].node),(yyvsp[-5].node));  }
#line 3092 "y.tab.c"
    break;

  case 158: /* iteration_cf: T_FORALL '(' id T_IN id '.' proc_call filterExpr ')' blockstatements  */
#line 455 "lrparser.y"
                                                                                       { 
																				(yyval.node)=Util::createNodeForForAllStmt((yyvsp[-7].node),(yyvsp[-5].node),(yyvsp[-3].node),(yyvsp[-2].node),(yyvsp[0].node),true);}
#line 3099 "y.tab.c"
    break;

  case 159: /* iteration_cf: T_FORALL '(' id T_IN leftSide ')' blockstatements  */
#line 457 "lrparser.y"
                                                                        { (yyval.node)=Util::createNodeForForStmt((yyvsp[-4].node),(yyvsp[-2].node),(yyvsp[0].node),true);}
#line 3105 "y.tab.c"
    break;

  case 160: /* iteration_cf: T_FOR '(' id T_IN leftSide ')' blockstatements  */
#line 458 "lrparser.y"
                                                                 { (yyval.node)=Util::createNodeForForStmt((yyvsp[-4].node),(yyvsp[-2].node),(yyvsp[0].node),false);}
#line 3111 "y.tab.c"
    break;

  case 161: /* iteration_cf: T_FOR '(' id T_IN id '.' proc_call filterExpr ')' blockstatements  */
#line 459 "lrparser.y"
                                                                                    {(yyval.node)=Util::createNodeForForAllStmt((yyvsp[-7].node),(yyvsp[-5].node),(yyvsp[-3].node),(yyvsp[-2].node),(yyvsp[0].node),false);}
#line 3117 "y.tab.c"
    break;

  case 162: /* iteration_cf: T_FOR '(' id T_IN indexExpr ')' blockstatements  */
#line 460 "lrparser.y"
                                                                  {(yyval.node) = Util::createNodeForForStmt((yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node), false);}
#line 3123 "y.tab.c"
    break;

  case 163: /* iteration_cf: T_FORALL '(' id T_IN indexExpr ')' blockstatements  */
#line 461 "lrparser.y"
                                                                     {(yyval.node) = Util::createNodeForForStmt((yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node), true);}
#line 3129 "y.tab.c"
    break;

  case 164: /* iteration_cf: T_LOOP '(' id T_IN expression T_TO expression T_BY expression ')' blockstatements  */
#line 462 "lrparser.y"
                                                                                                    {(yyval.node) = Util::createNodeForLoopStmt((yyvsp[-8].node), (yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node));}
#line 3135 "y.tab.c"
    break;

  case 165: /* iteration_cf: T_FOR '(' primitive id '=' rhs ';' boolean_expr ';' expression ')' blockstatements  */
#line 463 "lrparser.y"
                                                                                                     {(yyval.node) = Util::createNodeForSimpleForStmt((yyvsp[-9].node), (yyvsp[-8].node), (yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node)); }
#line 3141 "y.tab.c"
    break;

  case 166: /* filterExpr: %empty  */
#line 465 "lrparser.y"
                      { (yyval.node)=NULL;}
#line 3147 "y.tab.c"
    break;

  case 167: /* filterExpr: '.' T_FILTER '(' boolean_expr ')'  */
#line 466 "lrparser.y"
                                              { (yyval.node)=(yyvsp[-1].node);}
#line 3153 "y.tab.c"
    break;

  case 168: /* boolean_expr: expression  */
#line 468 "lrparser.y"
                          { (yyval.node)=(yyvsp[0].node) ;}
#line 3159 "y.tab.c"
    break;

  case 169: /* selection_cf: T_IF '(' boolean_expr ')' statement  */
#line 470 "lrparser.y"
                                                   { (yyval.node)=Util::createNodeForIfStmt((yyvsp[-2].node),(yyvsp[0].node),NULL); }
#line 3165 "y.tab.c"
    break;

  case 170: /* selection_cf: T_IF '(' boolean_expr ')' statement T_ELSE statement  */
#line 471 "lrparser.y"
                                                                           {(yyval.node)=Util::createNodeForIfStmt((yyvsp[-4].node),(yyvsp[-2].node),(yyvsp[0].node)); }
#line 3171 "y.tab.c"
    break;

  case 171: /* reduction: leftSide '=' reductionCall  */
#line 474 "lrparser.y"
                                       { (yyval.node)=Util::createNodeForReductionStmt((yyvsp[-2].node),(yyvsp[0].node)) ;}
#line 3177 "y.tab.c"
    break;

  case 172: /* reduction: '<' leftList '>' '=' '<' reductionCall ',' rightList '>'  */
#line 475 "lrparser.y"
                                                                              { reductionCall* reduc=(reductionCall*)(yyvsp[-3].node);
		                                                               (yyval.node)=Util::createNodeForReductionStmtList((yyvsp[-7].nodeList)->ASTNList,reduc,(yyvsp[-1].nodeList)->ASTNList);}
#line 3184 "y.tab.c"
    break;

  case 173: /* reduction: leftSide reduce_op expression  */
#line 477 "lrparser.y"
                                                   {(yyval.node)=Util::createNodeForReductionOpStmt((yyvsp[-2].node),(yyvsp[-1].ival),(yyvsp[0].node));}
#line 3190 "y.tab.c"
    break;

  case 174: /* reduction: expression reduce_op expression  */
#line 478 "lrparser.y"
                                         {printf ("here calling creation for red op\n") ;(yyval.node)=Util::createNodeForReductionOpStmt ((yyvsp[-2].node),(yyvsp[-1].ival),(yyvsp[0].node));}
#line 3196 "y.tab.c"
    break;

  case 175: /* reduce_op: T_ADD_ASSIGN  */
#line 481 "lrparser.y"
                         {(yyval.ival)=OPERATOR_ADDASSIGN;}
#line 3202 "y.tab.c"
    break;

  case 176: /* reduce_op: T_MUL_ASSIGN  */
#line 482 "lrparser.y"
                         {(yyval.ival)=OPERATOR_MULASSIGN;}
#line 3208 "y.tab.c"
    break;

  case 177: /* reduce_op: T_OR_ASSIGN  */
#line 483 "lrparser.y"
                                 {(yyval.ival)=OPERATOR_ORASSIGN;}
#line 3214 "y.tab.c"
    break;

  case 178: /* reduce_op: T_AND_ASSIGN  */
#line 484 "lrparser.y"
                                 {(yyval.ival)=OPERATOR_ANDASSIGN;}
#line 3220 "y.tab.c"
    break;

  case 179: /* reduce_op: T_SUB_ASSIGN  */
#line 485 "lrparser.y"
                                 {(yyval.ival)=OPERATOR_SUBASSIGN;}
#line 3226 "y.tab.c"
    break;

  case 180: /* leftList: leftSide ',' leftList  */
#line 487 "lrparser.y"
                                  { (yyval.nodeList)=Util::addToNList((yyvsp[0].nodeList),(yyvsp[-2].node));
                                         }
#line 3233 "y.tab.c"
    break;

  case 181: /* leftList: leftSide  */
#line 489 "lrparser.y"
                           { (yyval.nodeList)=Util::createNList((yyvsp[0].node));;}
#line 3239 "y.tab.c"
    break;

  case 182: /* rightList: val ',' rightList  */
#line 491 "lrparser.y"
                              { (yyval.nodeList)=Util::addToNList((yyvsp[0].nodeList),(yyvsp[-2].node));}
#line 3245 "y.tab.c"
    break;

  case 183: /* rightList: leftSide ',' rightList  */
#line 492 "lrparser.y"
                                   { ASTNode* node = Util::createNodeForId((yyvsp[-2].node));
			                         (yyval.nodeList)=Util::addToNList((yyvsp[0].nodeList),node);}
#line 3252 "y.tab.c"
    break;

  case 184: /* rightList: val  */
#line 494 "lrparser.y"
                   { (yyval.nodeList)=Util::createNList((yyvsp[0].node));}
#line 3258 "y.tab.c"
    break;

  case 185: /* rightList: leftSide  */
#line 495 "lrparser.y"
                              { ASTNode* node = Util::createNodeForId((yyvsp[0].node));
			            (yyval.nodeList)=Util::createNList(node);}
#line 3265 "y.tab.c"
    break;

  case 186: /* reductionCall: reduction_calls '(' arg_list ')'  */
#line 504 "lrparser.y"
                                                 {(yyval.node)=Util::createNodeforReductionCall((yyvsp[-3].ival),(yyvsp[-1].aList)->AList);}
#line 3271 "y.tab.c"
    break;

  case 187: /* reduction_calls: T_SUM  */
#line 506 "lrparser.y"
                        { (yyval.ival)=REDUCE_SUM;}
#line 3277 "y.tab.c"
    break;

  case 188: /* reduction_calls: T_COUNT  */
#line 507 "lrparser.y"
                           {(yyval.ival)=REDUCE_COUNT;}
#line 3283 "y.tab.c"
    break;

  case 189: /* reduction_calls: T_PRODUCT  */
#line 508 "lrparser.y"
                             {(yyval.ival)=REDUCE_PRODUCT;}
#line 3289 "y.tab.c"
    break;

  case 190: /* reduction_calls: T_MAX  */
#line 509 "lrparser.y"
                         {(yyval.ival)=REDUCE_MAX;}
#line 3295 "y.tab.c"
    break;

  case 191: /* reduction_calls: T_MIN  */
#line 510 "lrparser.y"
                         {(yyval.ival)=REDUCE_MIN;}
#line 3301 "y.tab.c"
    break;

  case 192: /* leftSide: id  */
#line 512 "lrparser.y"
              { (yyval.node)=(yyvsp[0].node); }
#line 3307 "y.tab.c"
    break;

  case 193: /* leftSide: oid  */
#line 513 "lrparser.y"
               { printf("Here hello \n"); (yyval.node)=(yyvsp[0].node); }
#line 3313 "y.tab.c"
    break;

  case 194: /* leftSide: tid  */
#line 514 "lrparser.y"
               {(yyval.node) = (yyvsp[0].node); }
#line 3319 "y.tab.c"
    break;

  case 195: /* leftSide: indexExpr  */
#line 515 "lrparser.y"
                    {(yyval.node)=(yyvsp[0].node);}
#line 3325 "y.tab.c"
    break;

  case 196: /* arg_list: %empty  */
#line 518 "lrparser.y"
              {
                 argList* aList=new argList();
				 (yyval.aList)=aList;  }
#line 3333 "y.tab.c"
    break;

  case 197: /* arg_list: assignment ',' arg_list  */
#line 522 "lrparser.y"
                                         {argument* a1=new argument();
		                          assignment* assign=(assignment*)(yyvsp[-2].node);
		                     a1->setAssign(assign);
							 a1->setAssignFlag();
		                 //a1->assignExpr=(assignment*)$1;
						 // a1->assign=true;
						  (yyval.aList)=Util::addToAList((yyvsp[0].aList),a1);
						  /*
						  for(argument* arg:$$->AList)
						  {
							  printf("VALUE OF ARG %d",arg->getAssignExpr()); //rm for warnings
						  }
						  */ 
						  
                          }
#line 3353 "y.tab.c"
    break;

  case 198: /* arg_list: expression ',' arg_list  */
#line 539 "lrparser.y"
                                             {argument* a1=new argument();
		                                Expression* expr=(Expression*)(yyvsp[-2].node);
										a1->setExpression(expr);
										a1->setExpressionFlag();
						               // a1->expressionflag=true;
										 (yyval.aList)=Util::addToAList((yyvsp[0].aList),a1);
						                }
#line 3365 "y.tab.c"
    break;

  case 199: /* arg_list: expression  */
#line 546 "lrparser.y"
                            {argument* a1=new argument();
		                 Expression* expr=(Expression*)(yyvsp[0].node);
						 a1->setExpression(expr);
						a1->setExpressionFlag();
						  (yyval.aList)=Util::createAList(a1); }
#line 3375 "y.tab.c"
    break;

  case 200: /* arg_list: assignment  */
#line 551 "lrparser.y"
                            { argument* a1=new argument();
		                   assignment* assign=(assignment*)(yyvsp[0].node);
		                     a1->setAssign(assign);
							 a1->setAssignFlag();
						   (yyval.aList)=Util::createAList(a1);
						   }
#line 3386 "y.tab.c"
    break;

  case 201: /* bfs_abstraction: T_BFS '(' id T_IN id '.' proc_call T_FROM id ')' filterExpr blockstatements reverse_abstraction  */
#line 559 "lrparser.y"
                                                                                                                 {(yyval.node)=Util::createIterateInBFSNode((yyvsp[-10].node),(yyvsp[-8].node),(yyvsp[-6].node),(yyvsp[-4].node),(yyvsp[-2].node),(yyvsp[-1].node),(yyvsp[0].node)) ;}
#line 3392 "y.tab.c"
    break;

  case 202: /* bfs_abstraction: T_BFS '(' id T_IN id '.' proc_call T_FROM id ')' filterExpr blockstatements  */
#line 560 "lrparser.y"
                                                                                                      {(yyval.node)=Util::createIterateInBFSNode((yyvsp[-9].node),(yyvsp[-7].node),(yyvsp[-5].node),(yyvsp[-3].node),(yyvsp[-1].node),(yyvsp[0].node),NULL) ; }
#line 3398 "y.tab.c"
    break;

  case 203: /* reverse_abstraction: T_REVERSE blockstatements  */
#line 564 "lrparser.y"
                                                 {(yyval.node)=Util::createIterateInReverseBFSNode(NULL,(yyvsp[0].node));}
#line 3404 "y.tab.c"
    break;

  case 204: /* reverse_abstraction: T_REVERSE '(' boolean_expr ')' blockstatements  */
#line 565 "lrparser.y"
                                                                       {(yyval.node)=Util::createIterateInReverseBFSNode((yyvsp[-2].node),(yyvsp[0].node));}
#line 3410 "y.tab.c"
    break;

  case 205: /* oid: id '.' id  */
#line 568 "lrparser.y"
                 { //Identifier* id1=(Identifier*)Util::createIdentifierNode($1);
                  // Identifier* id2=(Identifier*)Util::createIdentifierNode($1);
				   (yyval.node) = Util::createPropIdNode((yyvsp[-2].node),(yyvsp[0].node));
				    }
#line 3419 "y.tab.c"
    break;

  case 206: /* oid: id '.' id '[' id ']'  */
#line 572 "lrparser.y"
                                { ASTNode* expr1 = Util::createNodeForId((yyvsp[-3].node));
	                          ASTNode* expr2 = Util::createNodeForId((yyvsp[-1].node));
							  ASTNode* indexexpr =  Util::createNodeForIndexExpr(expr1, expr2, OPERATOR_INDEX);
	                          (yyval.node) = Util::createPropIdNode((yyvsp[-5].node) , indexexpr);}
#line 3428 "y.tab.c"
    break;

  case 207: /* tid: id '.' id '.' id  */
#line 579 "lrparser.y"
                       {// Identifier* id1=(Identifier*)Util::createIdentifierNode($1);
                  // Identifier* id2=(Identifier*)Util::createIdentifierNode($1);
				   (yyval.node)=Util::createPropIdNode((yyvsp[-4].node),(yyvsp[-2].node));
				    }
#line 3437 "y.tab.c"
    break;

  case 208: /* id: ID  */
#line 583 "lrparser.y"
          { 
	         (yyval.node)=Util::createIdentifierNode((yyvsp[0].text));  

            
            }
#line 3447 "y.tab.c"
    break;


#line 3451 "y.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 591 "lrparser.y"



void yyerror(const char *s) {
    fprintf(stderr, "%s\n", s);
}


int main(int argc,char **argv) 
{
  
  if(argc<4){
    std::cout<< "Usage: " << argv[0] << " [-s|-d] -f <dsl.sp> -b [cuda|omp|mpi|acc|multigpu|amd|hip|sycl] " << '\n';
    std::cout<< "E.g. : " << argv[0] << " -s -f ../graphcode/staticDSLCodes/sssp_dslV3 -b omp " << '\n';
    exit(-1);
  }
  
    //dsl_cpp_generator cpp_backend;
    SymbolTableBuilder stBuilder;
     FILE    *fd;
     
  int opt;
  char* fileName=NULL;
  backendTarget = NULL;
  bool staticGen = false;
  bool dynamicGen = false;
  bool optimize = false;
  bool multiFunction = false;

  while ((opt = getopt(argc, argv, "smdf:b:o")) != -1) 
  {
     switch (opt) 
     {
      case 'f':
        fileName = optarg;
        break;
      case 'b':
        backendTarget = optarg;
        break;
      case 's':
	    staticGen = true;
		break;
	  case 'd':
	    dynamicGen = true;
        break;	
	  case 'o':
	  	optimize = true;
		break;	
	  case 'm':
	  	multiFunction = true;
		break;
      case '?':
        fprintf(stderr,"Unknown option: %c\n", optopt);
		exit(-1);
        break;
      case ':':
        fprintf(stderr,"Missing arg for %c\n", optopt);
		exit(-1);
        break;
     }
  }
   
   printf("fileName %s\n",fileName);
   printf("Backend Target %s\n",backendTarget);

   
   if(fileName==NULL||backendTarget==NULL)
   {
	   if(fileName==NULL)
	      fprintf(stderr,"FileName option Error!\n");
	   if(backendTarget==NULL)
	      fprintf(stderr,"backendTarget option Error!\n")	;
	   exit(-1);	    
   }
   else
    {

		if(!((strcmp(backendTarget,"hip")==0)||(strcmp(backendTarget,"omp")==0)|| (strcmp(backendTarget,"amd")==0) || (strcmp(backendTarget,"mpi")==0)||(strcmp(backendTarget,"cuda")==0) || (strcmp(backendTarget,"acc")==0) || (strcmp(backendTarget,"sycl")==0)|| (strcmp(backendTarget,"multigpu")==0)))

		   {
			  fprintf(stderr, "Specified backend target is not implemented in the current version!\n");
			   exit(-1);
		   }
	}

   if(!(staticGen || dynamicGen)) {
		fprintf(stderr, "Type of graph(static/dynamic) not specified!\n");
		exit(-1);
     }
	  
     


   yyin= fopen(fileName,"r");
   
   if(!yyin) {
	printf("file doesn't exists!\n");
	exit(-1);
   }
   
   
   int error=yyparse();
   printf("Parsing done\n");
   if(error!=0)
   {
	   fprintf(stderr,"Parsing Error!\n");
	   exit(-1);
   }


	if(error!=1)
	{
     //TODO: redirect to different backend generator after comparing with the 'b' option
    std::cout << "at 1" << std::endl;
	stBuilder.buildST(frontEndContext.getFuncList());
	frontEndContext.setDynamicLinkFuncs(stBuilder.getDynamicLinkedFuncs());
	std::cout << "at 2" << std::endl;

	if(staticGen)
	  {
		  /*
		  if(optimize)
		  {
			  attachPropAnalyser apAnalyser;
			  apAnalyser.analyse(frontEndContext.getFuncList());

			  dataRaceAnalyser drAnalyser;
			  drAnalyser.analyse(frontEndContext.getFuncList());
			  
			  if(strcmp(backendTarget,"cuda")==0)
			  {
			  	deviceVarsAnalyser dvAnalyser;
				//cpp_backend.setOptimized();
			  	dvAnalyser.analyse(frontEndContext.getFuncList());
			  }
		  }
		  */
	  //cpp_backend.setFileName(fileName);
	  //cpp_backend.generate();
     if (strcmp(backendTarget, "cuda") == 0) {
        spcuda::dsl_cpp_generator cpp_backend;
        cpp_backend.setFileName(fileName);
	//~ cpp_backend.setOptimized();
		if (optimize) {
		attachPropAnalyser apAnalyser;
		apAnalyser.analyse(frontEndContext.getFuncList());

		dataRaceAnalyser drAnalyser;
		drAnalyser.analyse(frontEndContext.getFuncList());

		deviceVarsAnalyser dvAnalyser;
		dvAnalyser.analyse(frontEndContext.getFuncList());

		cpp_backend.setOptimized();
		}

		if(multiFunction){
			callGraphAnalyser cgAnalyser;	
			cgAnalyser.analyse(frontEndContext.getFuncList());

			cudaGlobalVariablesAnalyser cudaGlobalVarsAnalyser;
			cudaGlobalVarsAnalyser.analyse(frontEndContext.getFuncList());
			cpp_backend.setMultiFunction();
		}
		  
        cpp_backend.generate();
      }
      else if (strcmp(backendTarget, "omp") == 0) {
        spomp::dsl_cpp_generator cpp_backend;
	std::cout<< "size:" << frontEndContext.getFuncList().size() << '\n';
        cpp_backend.setFileName(fileName);
        cpp_backend.generate();
      }
	  else if (strcmp(backendTarget, "hip") == 0) {
        sphip::DslCppGenerator cpp_backend(fileName, 8);
	std::cout<< "size:" << frontEndContext.getFuncList().size() << '\n';
        cpp_backend.Generate();
      }
	  else if (strcmp(backendTarget, "mpi") == 0) {
        spmpi::dsl_cpp_generator cpp_backend;
		std::cout<< "size:" << frontEndContext.getFuncList().size() << '\n';
        cpp_backend.setFileName(fileName);
        cpp_backend.generate();
      } 
      else if (strcmp(backendTarget, "acc") == 0) {
        spacc::dsl_cpp_generator cpp_backend;
        cpp_backend.setFileName(fileName);
		if(optimize) {
			cpp_backend.setOptimized();
			blockVarsAnalyser bvAnalyser;
			bvAnalyser.analyse(frontEndContext.getFuncList());
		}
        cpp_backend.generate();
      }
	  else if(strcmp(backendTarget, "multigpu") == 0){
		spmultigpu::dsl_cpp_generator cpp_backend;
		pushpullAnalyser pp;
		pp.analyse(frontEndContext.getFuncList());
		cpp_backend.setFileName(fileName);
		cpp_backend.generate();
}
	  else if (strcmp(backendTarget, "sycl") == 0) {
		std::cout<<"GENERATING SYCL CODE"<<std::endl;
        spsycl::dsl_cpp_generator cpp_backend;
        cpp_backend.setFileName(fileName);
        cpp_backend.generate();
	  }
	  else if (strcmp(backendTarget, "amd") == 0) {
		std::cout<<"GENERATING OPENCL CODE"<<std::endl;
        spamd::dsl_cpp_generator cpp_backend;
        cpp_backend.setFileName(fileName);
        cpp_backend.generate();
	  }
      else
	    std::cout<< "invalid backend" << '\n';
	  }
	else 
	 {
		if(strcmp(backendTarget, "omp") == 0) {
		   spdynomp::dsl_dyn_cpp_generator cpp_dyn_gen;
		   cpp_dyn_gen.setFileName(fileName);
	       cpp_dyn_gen.generate();
		}
		if(strcmp(backendTarget, "mpi") == 0){
		   spdynmpi::dsl_dyn_cpp_generator cpp_dyn_gen;
		   std::cout<<"created dyn mpi"<<std::endl;
		   cpp_dyn_gen.setFileName(fileName);
		   std::cout<<"file name set"<<std::endl;
	       cpp_dyn_gen.generate();	
		}
	 }
	
   }

	printf("finished successfully\n");
   
   /* to generate code, ./finalcode -s/-d -f "filename" -b "backendname"*/
	return 0;   
	 
}
