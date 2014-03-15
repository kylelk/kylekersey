#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

struct node_rec {
         char *word;
         int freq;
         struct node_rec *left, *right;
};

typedef struct node_rec *tree_type;

tree_type root = NULL,
    pred[100];          /* list of predecessors;                        */
int parent,             /* index for the array of predecessors;         */
    DifferentWords = 0, /* counter of different words in text file;     */
    WordCnt = 0;        /* counter of all words in the same file;       */

#define UpdateGrandparentOrRoot { if (i > 0) /* if Q has grandparent */ \
                                        if (pred[i-1]->right == pred[i])\
                                                pred[i-1]->right = Q;   \
                                        else pred[i-1]->left  = Q;      \
                                  else root = Q;                        \
                                  parent--; /* grandparent becomes parent */  \
                                }
void
RotateR (tree_type Q, int i)
{
    pred[i]->left = Q->right;
    Q->right = pred[i];
    UpdateGrandparentOrRoot
}

void
RotateL (tree_type Q, int i)
{
    pred[i]->right = Q->left;
    Q->left = pred[i];
    UpdateGrandparentOrRoot
}

void
semisplay (tree_type Q)
{
    while (Q != root) {
        if (parent == 0)                  /* if Q's parent is the root; */
             if (pred[0]->left == Q)
                  RotateR(Q,0);
             else RotateL(Q,0);
        else if (pred[parent]->left == Q) /* if Q is left child;        */
             if (pred[parent-1]->left == pred[parent]) {
                  Q = pred[parent];
                  RotateR(pred[parent],parent-1);
                  parent--;
             }
             else {
                  RotateR(Q,parent);    /* rotate Q and its parent;     */
                  RotateL(Q,parent);    /* rotate Q and its new parent; */
             }
        else                            /* if Q is right child;         */
             if (pred[parent-1]->right == pred[parent]) {
                  Q = pred[parent];
                  RotateL(pred[parent],parent-1);
                  parent--;
             }
             else {
                  RotateL(Q,parent);    /* rotate Q and its parent;     */
                  RotateR(Q,parent);    /* rotate Q and its new parent; */
             }
        if (parent == -1)               /* update the root of the tree; */
            root = Q;
    }
}

void
CheckAndInsert (tree_type *root_addr, char *key)
{    tree_type p = *root_addr, prev = NULL, new_node;
     register int v;

     parent = -1;
     while (p) {
         prev = p;
         v = strcmp(p->word,key);
         if (v == 0) {          /* if key is in the tree,        */
              p->freq++;        /* update its frequency field,   */
              semisplay(p);     /* move it upwards;              */
              return;           /* and exit from CheckAndInsert; */
         }
         else if (v > 0)
              p = p->left;
         else p = p->right;
         pred[++parent] = prev; /* store prev in pred[]          */
     }                          /* the parent of p;              */
     if (!(new_node = (tree_type) malloc(sizeof(struct node_rec)))) {
         printf("No room for new nodes\n");
         exit(1);
     }                                        /* create a node for a new */
     new_node->left = new_node->right = NULL; /* word from text file,    */
     new_node->freq = 1;                      /* initialize its fields;  */
     if (!(new_node->word = (char*) malloc(strlen(key)+1))) {
        printf("No room for new words\n");
        exit(1);
     }
     strcpy(new_node->word,key);
     if (!*root_addr)           /* if tree is empty */
          *root_addr  = new_node;
     else if (v > 0)
          prev->left  = new_node;
     else prev->right = new_node;
}

void
inorder (tree_type p, FILE *FOut) /* transfer all words          */
{                                 /* from tree to an output file */
     if (p) {                     /* in alphabetical order;      */
         inorder (p->left,FOut);
         fprintf(FOut,"%s %d ", p->word,p->freq);
         DifferentWords++;
         WordCnt += p->freq;
         inorder (p->right,FOut);
     }
}


main()
{   char filename[15], s[50];
    register int ch = !EOF, i;
    FILE *FIn, *FOut;

    printf("Enter a file name: ");
    gets(filename);
    FOut = fopen ("splay.out","w");
    FIn  = fopen (filename,"r");
    while (ch != EOF) {
        while (ch != EOF && !isalnum(ch = fgetc(FIn)));/* skip non-letters */
        if (ch == EOF)
             break;
        for (s[0] = toupper(ch), i = 0; ch != EOF && isalnum(s[i]); i++)
             s[i+1] = toupper(ch = fgetc(FIn));
        s[i] = '\0';
        CheckAndInsert(&root,s);
    }
    inorder(root,FOut);
    fprintf(FOut,"\n\nFile %s contains %d words among which %d are different",
        filename,WordCnt,DifferentWords);
    fclose(FIn);
    fclose(FOut);
}
