int_format:
    .string "%d\n"

.globl asmfunc2
    .type asmfunc2, @function

asmfunc2:
    movl $123456, %eax

    # print content of %eax as decimal integer
    pusha           # save all registers
    pushl %eax
    pushl $int_format
    call printf
    add $8, %esp    # remove arguments from stack
    popa            # restore saved registers

    ret