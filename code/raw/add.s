; Set up activation frame pointer.

pushq   %rbp
movq    %rsp, %rbp

; Allocate space for activation frame local variables.

subq        $16, %rsp

; Set EAX=4 (32 bits.)

movl        $4, %eax

; Set CL=0 (8 bits.)

xorb        %cl, %cl

; Load effective address of "%dn" into RDX.

leaq        ?str(%rip), %rdx

; Copy that effective address now into RDI.

movq    %rdx, %rdi

; Move 4 into ESI, too.

movl        %eax, %esi

; Zero out EAX (CL=0, so now AL=0 though it was 4.)

movb    %cl, %al

; Call the printf() function to do its work.

callq       _printf

; Apparently gcc (or whatever compiler this is) feels that
; the return value for main() should be temporarily sitting
; on the local stack. Also, apparently, the hacker who wrote
; this C program DID NOT use a return statement to return
; a known value. So the local space was never set and so
; there is just garbage sitting there. So this instruction
; assumes that there was a smart programmer (not a cheap
; hacker) who actually used a "return 0" or something like that
; so that this value would be valid. This instruction now puts
; that value into EAX, where all "int" functions return their
; values when they return. Of course, this C-coder didn't
; provide a value so this following code is unnecessary and
; worse, as it returns garbage to the caller of main().

movl        -4(%rbp), %eax

; Now remove the local variable space. (First step to unwind
; the activation frame.)

addq    $16, %rsp

; Now restore the caller's activation frame. (Second step to
; unwind the activation frame.)

popq    %rbp

; Now return to the caller (which is some assembly code that
; proceeds to kill the process, etc.)