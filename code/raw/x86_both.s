/* x86_32 and x86_64 assembler example */
/* Copyright (C) 2007 Mario Lang <mlang@delysid.org> */
/* Compile with "gcc -nostdlib -o x86_both x86_both.S" adding -m32 or -m64 */
#ifdef __x86_64__
  #include <asm-x86_64/unistd.h>
#elif __i386__
  #include <asm-i386/unistd.h>
#else
  #error "Unhandled architecture"
#endif
STDOUT = 1

.data
#ifdef __x86_64__
program_name:		.string "X86_64 AT&T assembler example 1\n"
program_name_length   =	32
#else
program_name:		.string "X86 AT&T assembler example 1\n"
program_name_length   = 29
#endif
exit_code:		.long 0

.text
.globl _start
_start:
#ifdef __x86_64__
	movq $__NR_write, %rax
	movq $STDOUT, %rdi
	movq $program_name, %rsi
	movq $program_name_length, %rdx
	syscall
#elif __i386__
	movl $__NR_write, %eax
	movl $STDOUT, %ebx
	movl $program_name, %ecx
	movl $program_name_length, %edx
	int $0x80
#endif

#ifdef __x86_64__
	popq %rcx			# argc
#elif __i386__
	popl %ecx
#endif
argv:
#ifdef __x86_64__
	popq %rsi			# argv
	test %rsi, %rsi
	jz exit				# exit if last (NULL) argument string
#elif __i386__
	popl %ecx
	jecxz exit
#endif
#ifdef __x86_64__
	movq %rsi, %rdx
#elif __i386__
	movl %ecx, %ebx
	xorl %edx, %edx
#endif
strlen:
#ifdef __x86_64__
	lodsb
#elif __i386__
	movb (%ebx), %al
	inc %edx
	inc %ebx
#endif

	test %al, %al
	jnz strlen			# continue if not end of string

#ifdef __x86_64__
	movb $0x0A, -1(%rsi)		# replace NUL-byte with \n

	subq %rdx, %rsi			# calculate buffer size
	xchg %rdx, %rsi			# reorder for syscall conventions
	movq $__NR_write, %rax
	movq $STDOUT, %rdi		# file descriptor
	syscall
#elif __i386__
	movb $0x0A, -1(%ebx)           # replace NUL-byte with \n

	movl $__NR_write, %eax
	movl $STDOUT, %ebx
	int $0x80
#endif

	jmp argv			# process next argument

exit:
#ifdef __x86_64__
	movq $__NR_exit, %rax
	movl exit_code, %edi
	syscall
#elif __i386__
	movl $__NR_exit, %eax
	movl exit_code, %ebx
	int $0x80
#endif
