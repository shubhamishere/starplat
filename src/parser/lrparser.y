%define parse.trace
%{
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
%}


/* This is the yacc file in use for the DSL. The action part needs to be modified completely*/

%union {
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
     }
%token T_INT T_FLOAT T_BOOL T_DOUBLE T_STRING T_LONG  T_AUTOREF
%token T_FORALL T_FOR  T_P_INF  T_INF T_N_INF T_LOOP
%token T_FUNC T_IF T_ELSE T_WHILE T_RETURN T_DO T_IN T_FIXEDPOINT T_UNTIL T_FILTER T_TO T_BY
%token T_ADD_ASSIGN T_SUB_ASSIGN T_MUL_ASSIGN T_DIV_ASSIGN T_MOD_ASSIGN T_AND_ASSIGN T_XOR_ASSIGN
%token T_OR_ASSIGN T_INC_OP T_DEC_OP T_PTR_OP T_AND_OP T_OR_OP T_LE_OP T_GE_OP T_EQ_OP T_NE_OP T_ASTERISK
%token T_AND T_OR T_SUM T_AVG T_COUNT T_PRODUCT T_MAX T_MIN
%token T_GRAPH T_GNN T_DIR_GRAPH  T_NODE T_EDGE T_UPDATES T_CONTAINER T_POINT T_UNDIREDGE T_TRIANGLE T_NODEMAP T_VECTOR T_HASHMAP T_HASHSET T_BTREE T_GEOMCOMPLETEGRAPH T_GRAPH_LIST T_SET
%token T_NP  T_EP
%token T_LIST T_SET_NODES T_SET_EDGES  T_FROM T_RANDOMSHUFFLE T_ALLOCATE T_BREAK T_CONTINUE
%token T_BFS T_REVERSE
%token T_INCREMENTAL T_DECREMENTAL T_STATIC T_DYNAMIC
%token T_BATCH T_ONADD T_ONDELETE
%token return_func


%token <text> ID
%token <ival> INT_NUM
%token <fval> FLOAT_NUM
%token <bval> BOOL_VAL
%token <cval> CHAR_VAL
%token <text> STRING_VAL


%type <node> function_def function_data  return_func function_body param
%type <pList> paramList
%type <node> statement blockstatements assignment declaration proc_call control_flow reduction return_stmt batch_blockstmt on_add_blockstmt on_delete_blockstmt
%type <node> type type1 type2 type3
%type <node> primitive graph gnn collections structs property container nodemap vector hashmap hashset btree set
%type <node> id leftSide rhs expression oid val boolean_expr unary_expr indexExpr tid alloca_expr  
%type <node> bfs_abstraction filterExpr reverse_abstraction
%type <nodeList> leftList rightList
%type <node> iteration_cf selection_cf
%type <node> reductionCall break_stmt continue_stmt
%type <aList> arg_list
%type <ival> reduction_calls reduce_op
%type <bval> by_reference




 /* operator precedence
  * Lower is higher
  */
%left '?'
%left ':'
%left T_OR_OP
%left T_AND_OP
%left T_ADD_ASSIGN
%left T_EQ_OP  T_NE_OP
%left '<' '>'  T_LE_OP T_GE_OP
%left '+' '-' 
%left '*' '/' '%'

 

%%
program:  
        | program function_def {/* printf("LIST SIZE %d",frontEndContext.getFuncList().size())  ;*/ };

function_def: function_data  function_body  { 
	                                          Function* func=(Function*)$1;
                                              blockStatement* block=(blockStatement*)$2;
                                              func->setBlockStatement(block);
											   Util::addFuncToList(func);
											};

function_data: T_FUNC id '(' paramList ')' { 
										   $$=Util::createFuncNode($2,$4->PList);
                                           Util::setCurrentFuncType(GEN_FUNC);
										   Util::resetTemp(tempIds);
										   tempIds.clear();
	                                      };
			   | T_STATIC id '(' paramList ')' {
										   $$=Util::createStaticFuncNode($2,$4->PList);
                                            Util::setCurrentFuncType(STATIC_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
	                                      };
	           | T_INCREMENTAL '(' paramList ')' { 
										   $$=Util::createIncrementalNode($3->PList);
                                            Util::setCurrentFuncType(INCREMENTAL_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
	                                      };	
			   | T_DECREMENTAL '(' paramList ')' { 
										   $$=Util::createDecrementalNode($3->PList);
                                            Util::setCurrentFuncType(DECREMENTAL_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
	                                      };	
		       | T_DYNAMIC id '(' paramList ')' { $$=Util::createDynamicFuncNode($2,$4->PList);
                                            Util::setCurrentFuncType(DYNAMIC_FUNC);
											Util::resetTemp(tempIds);
											tempIds.clear();
											};	
			  // | return_func {$$ = $1};	

paramList: param {$$=Util::createPList($1);};
               | param ',' paramList {$$=Util::addToPList($3,$1); 
			                           };

type: type1 {$$ = $1;}
    | type2 {$$ = $1;}
	| type3 {$$ = $1;}

param : type1 by_reference id {  //Identifier* id=(Identifier*)Util::createIdentifierNode($3);
                        Type* type=(Type*)$1;
	                     Identifier* id=(Identifier*)$3;
						 
						 if(type->isGraphType())
						    {
							 tempIds.push_back(id);
						   
							}
						if(type->isGNNType())
							{
							tempIds.push_back(id);
						
							}

					printf("\n");
                    $$=Util::createParamNode($1,$2,$3); } ;
               | type2 by_reference id { // Identifier* id=(Identifier*)Util::createIdentifierNode($3);
			  
					
                             $$=Util::createParamNode($1,$2,$3);};
			   | type2 by_reference id '(' id ')' { // Identifier* id1=(Identifier*)Util::createIdentifierNode($5);
			                            //Identifier* id=(Identifier*)Util::createIdentifierNode($3);
				                        Type* tempType=(Type*)$1;
			                            if(tempType->isNodeEdgeType())
										  tempType->addSourceGraph($5);
				                         $$=Util::createParamNode(tempType,$2,$3);
									 };
				| type1 '&' id {
					Type* type = (Type*)$1;
					type->setRefType();
					Identifier* id=(Identifier*)$3;
					if(type->isGraphType())
					{
						tempIds.push_back(id);
					}
					printf("\n");
                    $$=Util::createParamNode($1,$3);
				}


by_reference : /* epsilon */ {$$ = false;};
		| '&' {$$ = true;};





function_body : blockstatements {$$=$1;};


statements :  {};
	| statements statement {printf ("found one statement\n") ; Util::addToBlock($2); };

statement: declaration ';'{$$=$1;};
	|assignment ';'{printf ("found an assignment type statement" ); $$=$1;};
	|proc_call ';' {printf ("found an proc call type statement" );$$=Util::createNodeForProcCallStmt($1);};
	|control_flow {printf ("found an control flow type statement" );$$=$1;};
	|reduction ';'{printf ("found an reduction type statement" );$$=$1;};
	| bfs_abstraction {printf ("found bfs\n") ;$$=$1; };
	| blockstatements {printf ("found block\n") ;$$=$1;};
	| unary_expr ';' {printf ("found unary\n") ;$$=Util::createNodeForUnaryStatements($1);};
	| return_stmt ';' {printf ("found return\n") ;$$ = $1 ;};
	| batch_blockstmt  {printf ("found batch\n") ;$$ = $1;};
	| on_add_blockstmt {printf ("found on add block\n") ;$$ = $1;};
	| on_delete_blockstmt {printf ("found delete block\n") ;$$ = $1;};
	| break_stmt ';' {printf ("found break\n") ;$$ = Util::createNodeForBreakStatement();};
	| continue_stmt ';' {printf ("found continue\n") ;$$ = Util::createNodeForContinueStatement();};

break_stmt : T_BREAK {printf ("found break\n") ;};
continue_stmt : T_CONTINUE {printf ("found continue\n") ;};

blockstatements : block_begin statements block_end { $$=Util::finishBlock();};

batch_blockstmt : T_BATCH '(' id ':' expression ')' blockstatements {$$ = Util::createBatchBlockStmt($3, $5, $7);};

on_add_blockstmt : T_ONADD '(' id T_IN id '.' proc_call ')' ':' blockstatements {$$ = Util::createOnAddBlock($3, $5, $7, $10);};

on_delete_blockstmt : T_ONDELETE '(' id T_IN id '.' proc_call ')' ':' blockstatements {$$ = Util::createOnDeleteBlock($3, $5, $7, $10);};

block_begin:'{' { Util::createNewBlock(); }

block_end:'}'

return_stmt : T_RETURN expression {$$ = Util::createReturnStatementNode($2);}
	| T_RETURN ';' {$$ = Util::createReturnStatementNode(NULL);};
               

declaration : type1 id   {
	                     Type* type=(Type*)$1;
	                     Identifier* id=(Identifier*)$2;
						 
						 if(type->isGraphType())
						    Util::storeGraphId(id);

                         $$=Util::createNormalDeclNode($1,$2);};
	| type1 id '=' rhs  {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                    
	                    $$=Util::createAssignedDeclNode($1,$2,$4);};
	| type2 id  {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	            
                         $$=Util::createNormalDeclNode($1,$2); };
	| type2 id '=' rhs {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    $$=Util::createAssignedDeclNode($1,$2,$4);};
	| type3 id '=' rhs {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    $$=Util::createAssignedDeclNode($1,$2,$4);};

	| type3 id '(' rhs ')' {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    $$=Util::createParamDeclNode($1,$2,$4);};

	| type2 id '(' rhs ')' {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    $$=Util::createParamDeclNode($1,$2,$4);};

	| type1 id '(' rhs ')' {//Identifier* id=(Identifier*)Util::createIdentifierNode($2);
	                   
	                    $$=Util::createParamDeclNode($1,$2,$4);};


type1: primitive {$$=$1; };
	| graph {$$=$1;};
	| collections { $$=$1;};
	| structs {$$=$1;};
	| type1 T_ASTERISK {
						Type* type=(Type*)$1;
						type->incrementPointerStarCount();
						$$=$1;
						};
	| gnn {$$=$1;}; 


primitive: T_INT { $$=Util::createPrimitiveTypeNode(TYPE_INT);};
	| T_FLOAT { $$=Util::createPrimitiveTypeNode(TYPE_FLOAT);};
	| T_BOOL { $$=Util::createPrimitiveTypeNode(TYPE_BOOL);};
	| T_DOUBLE { $$=Util::createPrimitiveTypeNode(TYPE_DOUBLE); };
    | T_LONG {$$=$$=Util::createPrimitiveTypeNode(TYPE_LONG);};
	| T_STRING {$$=$$=Util::createPrimitiveTypeNode(TYPE_STRING);};

type3: T_AUTOREF { $$=Util::createPrimitiveTypeNode(TYPE_AUTOREF);};	

graph : T_GRAPH { $$=Util::createGraphTypeNode(TYPE_GRAPH,NULL);};
	|T_DIR_GRAPH {$$=Util::createGraphTypeNode(TYPE_DIRGRAPH,NULL);};
	| T_GEOMCOMPLETEGRAPH { $$=Util::createGraphTypeNode(TYPE_GEOMCOMPLETEGRAPH,NULL);};
	| T_GRAPH_LIST { $$=Util::createGraphTypeNode(TYPE_GRAPH_LIST,NULL);};


gnn : T_GNN { $$=Util::createGNNTypeNode(TYPE_GNN,NULL);};

collections : T_LIST { $$=Util::createCollectionTypeNode(TYPE_LIST,NULL);};
		| T_SET_NODES '<' id '>' {//Identifier* id=(Identifier*)Util::createIdentifierNode($3);
			                     $$=Util::createCollectionTypeNode(TYPE_SETN,$3);};
        | T_SET_EDGES '<' id '>' {// Identifier* id=(Identifier*)Util::createIdentifierNode($3);
					                    $$=Util::createCollectionTypeNode(TYPE_SETE,$3);};
		| T_UPDATES '<' id '>'   { $$=Util::createCollectionTypeNode(TYPE_UPDATES,$3);}
	    | container {$$ = $1;}
      	| vector {$$ = $1;}
		| set    {$$ = $1;}
		| nodemap   {$$ = $1;}
		| hashmap {$$ = $1;}
	    | hashset {$$ = $1;}
		| btree {$$ = $1;}

structs: T_POINT { $$=Util::createPointTypeNode(TYPE_POINT);};
	| T_UNDIREDGE { $$=Util::createUndirectedEdgeTypeNode(TYPE_UNDIREDGE);};
	| T_TRIANGLE { $$=Util::createTriangleTypeNode(TYPE_TRIANGLE);};

container : T_CONTAINER '<' type '>' '(' arg_list ',' type ')' {$$ = Util::createContainerTypeNode(TYPE_CONTAINER, $3, $6->AList, $8);}
          | T_CONTAINER '<' type '>' '(' arg_list ')' { $$ =  Util::createContainerTypeNode(TYPE_CONTAINER, $3, $6->AList, NULL);}
          | T_CONTAINER '<' type '>' { list<argument*> argList;
			                          $$ = Util::createContainerTypeNode(TYPE_CONTAINER, $3, argList, NULL);}	

vector: T_VECTOR'<' type '>' '(' arg_list ',' type ')' {$$ = Util::createContainerTypeNode(TYPE_VECTOR, $3, $6->AList, $8);}
          | T_VECTOR'<' type '>' '(' arg_list ')' { $$ =  Util::createContainerTypeNode(TYPE_VECTOR, $3, $6->AList, NULL);}
          | T_VECTOR'<' type '>' { list<argument*> argList;
			                          $$ = Util::createContainerTypeNode(TYPE_VECTOR, $3, argList, NULL);}	
		  | T_VECTOR'<' type '>' '&' { list<argument*> argList;
			                          $$ = Util::createContainerTypeRefNode(TYPE_VECTOR, $3, argList, NULL);}	

set: T_SET'<' type '>' '(' arg_list ',' type ')' {$$ = Util::createContainerTypeNode(TYPE_SET, $3, $6->AList, $8);}
          | T_SET'<' type '>' '(' arg_list ')' { $$ =  Util::createContainerTypeNode(TYPE_SET, $3, $6->AList, NULL);}
          | T_SET'<' type '>' { list<argument*> argList;
			                          $$ = Util::createContainerTypeNode(TYPE_SET, $3, argList, NULL);}	
		   | T_SET'<' type '>' '&' { list<argument*> argList;
			                          $$ = Util::createContainerTypeRefNode(TYPE_SET, $3, argList, NULL);}		

nodemap : T_NODEMAP '(' type ')' {$$ = Util::createNodeMapTypeNode(TYPE_NODEMAP, $3);}

hashmap : T_HASHMAP '<' type ',' type '>' { list<argument*> argList;
			                          $$ = Util::createHashMapTypeNode(TYPE_HASHMAP, $3, argList, $5);}

hashset : T_HASHSET '<' type '>' { list<argument*> argList;
			                          $$ = Util::createHashSetTypeNode(TYPE_HASHSET, $3, argList, NULL);}

btree : T_BTREE { $$ = Util::createBtreeTypeNode(TYPE_BTREE);};

type2 : T_NODE {$$=Util::createNodeEdgeTypeNode(TYPE_NODE) ;};
       | T_EDGE {$$=Util::createNodeEdgeTypeNode(TYPE_EDGE);};
	   | property {$$=$1;};

property : T_NP '<' primitive '>' { $$=Util::createPropertyTypeNode(TYPE_PROPNODE,$3); };
              | T_EP '<' primitive '>' { $$=Util::createPropertyTypeNode(TYPE_PROPEDGE,$3); };
			  | T_NP '<' collections '>'{  $$=Util::createPropertyTypeNode(TYPE_PROPNODE,$3); };
			  | T_EP '<' collections '>' {$$=Util::createPropertyTypeNode(TYPE_PROPEDGE,$3);};
              | T_NP '<' T_NODE '>' {ASTNode* type = Util::createNodeEdgeTypeNode(TYPE_NODE);
			                         $$=Util::createPropertyTypeNode(TYPE_PROPNODE, type); }
			  | T_NP '<' T_EDGE '>' {ASTNode* type = Util::createNodeEdgeTypeNode(TYPE_EDGE);
			                         $$=Util::createPropertyTypeNode(TYPE_PROPNODE, type); }	

assignment :  leftSide '=' rhs  { printf("testassign\n");$$=Util::createAssignmentNode($1,$3);} ;
              | indexExpr '=' rhs {printf ("called assign for count\n") ; $$=Util::createAssignmentNode($1 , $3);};    
			  | id '=' expression  { $$ = Util::createAssignmentNode($1, $3); };    


rhs : expression { $$=$1;};

expression : proc_call { $$=$1;};
             | expression '+' expression { $$=Util::createNodeForArithmeticExpr($1,$3,OPERATOR_ADD);};
	         | expression '-' expression { $$=Util::createNodeForArithmeticExpr($1,$3,OPERATOR_SUB);};
	         | expression '*' expression {$$=Util::createNodeForArithmeticExpr($1,$3,OPERATOR_MUL);};
			 | expression T_ASTERISK expression {$$=Util::createNodeForArithmeticExpr($1,$3,OPERATOR_MUL);};
	         | expression'/' expression{$$=Util::createNodeForArithmeticExpr($1,$3,OPERATOR_DIV);};
			 | expression '%' expression{$$=Util::createNodeForArithmeticExpr($1,$3,OPERATOR_MOD);};
             | expression T_AND_OP expression {$$=Util::createNodeForLogicalExpr($1,$3,OPERATOR_AND);};
	         | expression T_OR_OP  expression {$$=Util::createNodeForLogicalExpr($1,$3,OPERATOR_OR);};
	         | expression T_LE_OP expression {$$=Util::createNodeForRelationalExpr($1,$3,OPERATOR_LE);};
	         | expression T_GE_OP expression{$$=Util::createNodeForRelationalExpr($1,$3,OPERATOR_GE);};
			 | expression '<' expression{$$=Util::createNodeForRelationalExpr($1,$3,OPERATOR_LT);};
			 | expression '>' expression{$$=Util::createNodeForRelationalExpr($1,$3,OPERATOR_GT);};
			 | expression T_EQ_OP expression{$$=Util::createNodeForRelationalExpr($1,$3,OPERATOR_EQ);};
             | expression T_NE_OP expression{$$=Util::createNodeForRelationalExpr($1,$3,OPERATOR_NE);};
			 | '!'expression {$$=Util::createNodeForUnaryExpr($2,OPERATOR_NOT);};
		     | '(' expression ')' { Expression* expr=(Expression*)$2;
				                     expr->setEnclosedBrackets();
			                        $$=expr;};
	         | val {$$=$1;};
			 | leftSide { $$=Util::createNodeForId($1);};
			 | unary_expr {$$=$1;};
			 | indexExpr {$$ = $1;};
			 | alloca_expr {$$= $1;};

alloca_expr : T_ALLOCATE '<' type '>' '(' arg_list ')' { 
					$$ = Util::createNodeForAllocaExpr($3, $6->AList); 
				};

indexExpr : expression '[' expression ']' {printf("first done this \n");$$ = Util::createNodeForIndexExpr($1, $3, OPERATOR_INDEX);};

unary_expr :   expression T_INC_OP {$$=Util::createNodeForUnaryExpr($1,OPERATOR_INC);};
			 |  expression T_DEC_OP {$$=Util::createNodeForUnaryExpr($1,OPERATOR_DEC);}; 			 

proc_call : leftSide '(' arg_list ')' { 
										ASTNode* proc_callId = $1;
										if(proc_callId->getTypeofNode()==NODE_ID){
											Identifier* id = (Identifier*)proc_callId;
											if(strcmp(id->getIdentifier(),"tsort")==0){
												frontEndContext.setThrustUsed(true);
											}
										}
                                       $$ = Util::createNodeForProcCall($1,$3->AList,NULL); 

									    };
			| T_INCREMENTAL '(' arg_list ')' { ASTNode* id = Util::createIdentifierNode("Incremental");
			                                   $$ = Util::createNodeForProcCall(id, $3->AList,NULL); 

				                               };
			| T_DECREMENTAL '(' arg_list ')' { ASTNode* id = Util::createIdentifierNode("Decremental");
			                                   $$ = Util::createNodeForProcCall(id, $3->AList,NULL); 

				                               };	
			| indexExpr '.' leftSide '(' arg_list ')' {
                                                   
													 Expression* expr = (Expression*)$1;
                                                     $$ = Util::createNodeForProcCall($3 , $5->AList, expr); 

									                 };								   							   
											   					
		



val : INT_NUM { $$ = Util::createNodeForIval($1); };
	| FLOAT_NUM {$$ = Util::createNodeForFval($1);};
	| BOOL_VAL { $$ = Util::createNodeForBval($1);};
	| STRING_VAL { $$ = Util::createNodeForSval($1);};
	| T_INF {$$=Util::createNodeForINF(true);};
	| T_P_INF {$$=Util::createNodeForINF(true);};
	| T_N_INF {$$=Util::createNodeForINF(false);};


control_flow : selection_cf { $$=$1; };
              | iteration_cf { $$=$1; };

iteration_cf : T_FIXEDPOINT T_UNTIL '(' id ':' expression ')' blockstatements { $$=Util::createNodeForFixedPointStmt($4,$6,$8);};
		   | T_WHILE '(' boolean_expr')' blockstatements {$$=Util::createNodeForWhileStmt($3,$5); };
		   | T_DO blockstatements T_WHILE '(' boolean_expr ')' ';' {$$=Util::createNodeForDoWhileStmt($5,$2);  };
		| T_FORALL '(' id T_IN id '.' proc_call filterExpr')'  blockstatements { 
																				$$=Util::createNodeForForAllStmt($3,$5,$7,$8,$10,true);};
		| T_FORALL '(' id T_IN leftSide ')' blockstatements	{ $$=Util::createNodeForForStmt($3,$5,$7,true);};																	
		| T_FOR '(' id T_IN leftSide ')' blockstatements { $$=Util::createNodeForForStmt($3,$5,$7,false);};
		| T_FOR '(' id T_IN id '.' proc_call  filterExpr')' blockstatements {$$=Util::createNodeForForAllStmt($3,$5,$7,$8,$10,false);};
		| T_FOR '(' id T_IN indexExpr ')' blockstatements {$$ = Util::createNodeForForStmt($3, $5, $7, false);};
		| T_FORALL '(' id T_IN indexExpr ')' blockstatements {$$ = Util::createNodeForForStmt($3, $5, $7, true);};
		| T_LOOP '(' id T_IN expression T_TO expression T_BY expression ')' blockstatements {$$ = Util::createNodeForLoopStmt($3, $5, $7, $9, $11);};
		| T_FOR '(' primitive id '=' rhs ';' boolean_expr ';' expression ')' blockstatements {$$ = Util::createNodeForSimpleForStmt($3, $4, $6, $8, $10, $12); };

filterExpr  :         { $$=NULL;};
            |'.' T_FILTER '(' boolean_expr ')'{ $$=$4;};

boolean_expr : expression { $$=$1 ;};

selection_cf : T_IF '(' boolean_expr ')' statement { $$=Util::createNodeForIfStmt($3,$5,NULL); }; 
	           | T_IF '(' boolean_expr ')' statement T_ELSE statement  {$$=Util::createNodeForIfStmt($3,$5,$7); };


reduction : leftSide '=' reductionCall { $$=Util::createNodeForReductionStmt($1,$3) ;}
		   |'<' leftList '>' '=' '<' reductionCall ',' rightList '>'  { reductionCall* reduc=(reductionCall*)$6;
		                                                               $$=Util::createNodeForReductionStmtList($2->ASTNList,reduc,$8->ASTNList);};
		   | leftSide reduce_op expression {$$=Util::createNodeForReductionOpStmt($1,$2,$3);}; 															   
       | expression reduce_op expression {printf ("here calling creation for red op\n") ;$$=Util::createNodeForReductionOpStmt ($1,$2,$3);};


reduce_op : T_ADD_ASSIGN {$$=OPERATOR_ADDASSIGN;};
          | T_MUL_ASSIGN {$$=OPERATOR_MULASSIGN;}
		  | T_OR_ASSIGN  {$$=OPERATOR_ORASSIGN;}
		  | T_AND_ASSIGN {$$=OPERATOR_ANDASSIGN;}
		  | T_SUB_ASSIGN {$$=OPERATOR_SUBASSIGN;}

leftList :  leftSide ',' leftList { $$=Util::addToNList($3,$1);
                                         };
		 | leftSide{ $$=Util::createNList($1);;};

rightList : val ',' rightList { $$=Util::addToNList($3,$1);};
          | leftSide ',' rightList { ASTNode* node = Util::createNodeForId($1);
			                         $$=Util::addToNList($3,node);};
          | val    { $$=Util::createNList($1);};
		  | leftSide  { ASTNode* node = Util::createNodeForId($1);
			            $$=Util::createNList(node);};

            /*reductionCall ',' val { $$=new tempNode();
	                                $$->reducCall=(reductionCall*)$1;
                                    $$->exprVal=(Expression*)$3; };
          | reductionCall { 
			                $$->reducCall=(reductionCall*)$1;} ;*/

reductionCall : reduction_calls '(' arg_list ')' {$$=Util::createNodeforReductionCall($1,$3->AList);} ;

reduction_calls : T_SUM { $$=REDUCE_SUM;};
	         | T_COUNT {$$=REDUCE_COUNT;};
	         | T_PRODUCT {$$=REDUCE_PRODUCT;};
	         | T_MAX {$$=REDUCE_MAX;};
	         | T_MIN {$$=REDUCE_MIN;};

leftSide : id { $$=$1; };
         | oid { printf("Here hello \n"); $$=$1; };
         | tid {$$ = $1; };	
         | indexExpr{$$=$1;};
		  

arg_list :    {
                 argList* aList=new argList();
				 $$=aList;  };
		      
		|assignment ',' arg_list {argument* a1=new argument();
		                          assignment* assign=(assignment*)$1;
		                     a1->setAssign(assign);
							 a1->setAssignFlag();
		                 //a1->assignExpr=(assignment*)$1;
						 // a1->assign=true;
						  $$=Util::addToAList($3,a1);
						  /*
						  for(argument* arg:$$->AList)
						  {
							  printf("VALUE OF ARG %d",arg->getAssignExpr()); //rm for warnings
						  }
						  */ 
						  
                          };


	       |   expression ',' arg_list   {argument* a1=new argument();
		                                Expression* expr=(Expression*)$1;
										a1->setExpression(expr);
										a1->setExpressionFlag();
						               // a1->expressionflag=true;
										 $$=Util::addToAList($3,a1);
						                };
	       | expression {argument* a1=new argument();
		                 Expression* expr=(Expression*)$1;
						 a1->setExpression(expr);
						a1->setExpressionFlag();
						  $$=Util::createAList(a1); };
	       | assignment { argument* a1=new argument();
		                   assignment* assign=(assignment*)$1;
		                     a1->setAssign(assign);
							 a1->setAssignFlag();
						   $$=Util::createAList(a1);
						   };


bfs_abstraction	: T_BFS '(' id T_IN id '.' proc_call T_FROM id ')' filterExpr blockstatements reverse_abstraction{$$=Util::createIterateInBFSNode($3,$5,$7,$9,$11,$12,$13) ;};
			| T_BFS '(' id T_IN id '.' proc_call T_FROM id ')' filterExpr blockstatements {$$=Util::createIterateInBFSNode($3,$5,$7,$9,$11,$12,NULL) ; };



reverse_abstraction :  T_REVERSE blockstatements {$$=Util::createIterateInReverseBFSNode(NULL,$2);};
                     | T_REVERSE '(' boolean_expr ')'  blockstatements {$$=Util::createIterateInReverseBFSNode($3,$5);};


oid :  id '.' id { //Identifier* id1=(Identifier*)Util::createIdentifierNode($1);
                  // Identifier* id2=(Identifier*)Util::createIdentifierNode($1);
				   $$ = Util::createPropIdNode($1,$3);
				    };	
	 | id '.' id '[' id ']' { ASTNode* expr1 = Util::createNodeForId($3);
	                          ASTNode* expr2 = Util::createNodeForId($5);
							  ASTNode* indexexpr =  Util::createNodeForIndexExpr(expr1, expr2, OPERATOR_INDEX);
	                          $$ = Util::createPropIdNode($1 , indexexpr);};					
    				
					 

tid : id '.' id '.' id {// Identifier* id1=(Identifier*)Util::createIdentifierNode($1);
                  // Identifier* id2=(Identifier*)Util::createIdentifierNode($1);
				   $$=Util::createPropIdNode($1,$3);
				    };
id : ID   { 
	         $$=Util::createIdentifierNode($1);  

            
            };                                                   
          


%%


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

		if(!((strcmp(backendTarget,"hip")==0)||(strcmp(backendTarget,"omp")==0)|| (strcmp(backendTarget,"amd")==0) || (strcmp(backendTarget,"mpi")==0)||(strcmp(backendTarget,"cuda")==0) || (strcmp(backendTarget,"acc")==0) || (strcmp(backendTarget,"sycl")==0)|| (strcmp(backendTarget,"multigpu")==0) || (strcmp(backendTarget,"webgpu")==0)))

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
      else if (strcmp(backendTarget, "webgpu") == 0) {
        std::cout << "[DEBUG] Entering WebGPU backend section" << std::endl;
        const auto& funcList = frontEndContext.getFuncList();
        std::cout << "[DEBUG] Got funcList, size: " << funcList.size() << std::endl;
        if (funcList.empty()) {
            std::cerr << "[WebGPU] Error: Function list is empty!" << std::endl;
        } else {
            std::cout << "[DEBUG] Creating WebGPU generator" << std::endl;
            spwebgpu::dsl_webgpu_generator webgpu_backend;
            std::string outFile = "../graphcode/generated_webgpu/output.js";
            std::cout << "[DEBUG] Calling generate with funcList.front()=" << funcList.front() << std::endl;
            webgpu_backend.generate(funcList.front(), outFile);
        }
      }
      else
	    std::cout<< "invalid backend" << '\n';
	  }
	// Dynamic graph backends removed - focusing on static graphs only
	
   }

	printf("finished successfully\n");
   
   /* to generate code, ./finalcode -s/-d -f "filename" -b "backendname"*/
	return 0;   
	 
}
