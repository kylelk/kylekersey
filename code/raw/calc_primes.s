	.section	__TEXT,__text,regular,pure_instructions
	.globl	_is_prime
	.align	4, 0x90
_is_prime:
Leh_func_begin1:
	pushq	%rbp
Ltmp0:
	movq	%rsp, %rbp
Ltmp1:
	subq	$32, %rsp
Ltmp2:
	movl	%edi, -4(%rbp)
	movl	$3, -16(%rbp)
	movl	-4(%rbp), %eax
	cvtsi2sd	%eax, %xmm0
	callq	_sqrt
	cvttsd2si	%xmm0, %eax
	movl	%eax, -20(%rbp)
	movl	-4(%rbp), %eax
	cmpl	$2, %eax
	jg	LBB1_2
	movl	$1, -12(%rbp)
	jmp	LBB1_10
LBB1_2:
	movl	-4(%rbp), %eax
	andl	$1, %eax
	cmpl	$0, %eax
	jne	LBB1_4
	movl	$0, -12(%rbp)
	jmp	LBB1_10
LBB1_4:
	movl	$3, -16(%rbp)
	jmp	LBB1_8
LBB1_5:
	movl	-4(%rbp), %eax
	movl	-16(%rbp), %ecx
	cltd
	idivl	%ecx
	testl	%edx, %edx
	jne	LBB1_7
	movl	$0, -12(%rbp)
	jmp	LBB1_10
LBB1_7:
	movl	-16(%rbp), %eax
	addl	$2, %eax
	movl	%eax, -16(%rbp)
LBB1_8:
	movl	-16(%rbp), %eax
	movl	-20(%rbp), %ecx
	cmpl	%ecx, %eax
	jle	LBB1_5
	movl	$1, -12(%rbp)
LBB1_10:
	movl	-12(%rbp), %eax
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	ret
Leh_func_end1:

	.globl	_main
	.align	4, 0x90
_main:
Leh_func_begin2:
	pushq	%rbp
Ltmp3:
	movq	%rsp, %rbp
Ltmp4:
	subq	$32, %rsp
Ltmp5:
	movl	%edi, -4(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$1, -28(%rbp)
	movl	-4(%rbp), %eax
	cmpl	$1, %eax
	jg	LBB2_2
	movl	$1, -24(%rbp)
	jmp	LBB2_8
LBB2_2:
	movl	$1, -28(%rbp)
	jmp	LBB2_6
LBB2_3:
	movl	-28(%rbp), %eax
	movl	%eax, %edi
	callq	_is_prime
	movl	%eax, %ecx
	cmpl	$0, %ecx
	je	LBB2_5
	movl	-28(%rbp), %eax
	xorb	%cl, %cl
	leaq	L_.str(%rip), %rdx
	movq	%rdx, %rdi
	movl	%eax, %esi
	movb	%cl, %al
	callq	_printf
LBB2_5:
	movl	-28(%rbp), %eax
	addl	$2, %eax
	movl	%eax, -28(%rbp)
LBB2_6:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	callq	_atoi
	movl	%eax, %ecx
	movl	-28(%rbp), %edx
	cmpl	%edx, %ecx
	jg	LBB2_3
	movl	$0, -24(%rbp)
LBB2_8:
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	ret
Leh_func_end2:

	.section	__TEXT,__cstring,cstring_literals
L_.str:
	.asciz	 "%d\n"

	.section	__TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame0:
Lsection_eh_frame:
Leh_frame_common:
Lset0 = Leh_frame_common_end-Leh_frame_common_begin
	.long	Lset0
Leh_frame_common_begin:
	.long	0
	.byte	1
	.asciz	 "zR"
	.byte	1
	.byte	120
	.byte	16
	.byte	1
	.byte	16
	.byte	12
	.byte	7
	.byte	8
	.byte	144
	.byte	1
	.align	3
Leh_frame_common_end:
	.globl	_is_prime.eh
_is_prime.eh:
Lset1 = Leh_frame_end1-Leh_frame_begin1
	.long	Lset1
Leh_frame_begin1:
Lset2 = Leh_frame_begin1-Leh_frame_common
	.long	Lset2
Ltmp6:
	.quad	Leh_func_begin1-Ltmp6
Lset3 = Leh_func_end1-Leh_func_begin1
	.quad	Lset3
	.byte	0
	.byte	4
Lset4 = Ltmp0-Leh_func_begin1
	.long	Lset4
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset5 = Ltmp1-Ltmp0
	.long	Lset5
	.byte	13
	.byte	6
	.align	3
Leh_frame_end1:

	.globl	_main.eh
_main.eh:
Lset6 = Leh_frame_end2-Leh_frame_begin2
	.long	Lset6
Leh_frame_begin2:
Lset7 = Leh_frame_begin2-Leh_frame_common
	.long	Lset7
Ltmp7:
	.quad	Leh_func_begin2-Ltmp7
Lset8 = Leh_func_end2-Leh_func_begin2
	.quad	Lset8
	.byte	0
	.byte	4
Lset9 = Ltmp3-Leh_func_begin2
	.long	Lset9
	.byte	14
	.byte	16
	.byte	134
	.byte	2
	.byte	4
Lset10 = Ltmp4-Ltmp3
	.long	Lset10
	.byte	13
	.byte	6
	.align	3
Leh_frame_end2:


.subsections_via_symbols
