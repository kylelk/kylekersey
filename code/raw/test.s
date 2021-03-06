extern printf
	global main
 
	section .text
main
	mov	eax, dword [_a]
	mov	ecx, dword [_b]
	push	ecx
	push	eax
 
	and 	eax, ecx
	mov	ebx, _opand
	call	out_ops
 
	call	get_nums
	or	eax, ecx
	mov	ebx, _opor
	call	out_ops
 
	call	get_nums
	xor     eax, ecx
	mov	ebx, _opxor
	call	out_ops
 
	call	get_nums
	shr	eax, cl
	mov	ebx, _opshr
	call	out_ops
 
	call	get_nums
	shl	eax, cl
	mov	ebx, _opshl
	call	out_ops
 
	call	get_nums
	rol	eax, cl
	mov	ebx, _oprol
	call	out_ops
 
	call	get_nums
	ror	eax, cl
	mov	ebx, _opror
	call	out_ops
 
	call	get_nums
	sal	eax, cl
	mov	ebx, _opsal
	call	out_ops
 
	call	get_nums
	sar	eax, cl
	mov	ebx, _opsar
	call	out_ops
 
	mov	eax, dword [esp+0]
	not	eax
	push 	eax
	not	eax
	push	eax
	push	_opnot
	push	_null
	push	_testn
	call	printf
	add	esp, 20
 
	add	esp, 8
	ret
 
out_ops
	push	eax
	push	ecx
	push	ebx
	push	dword [_a]
	push	_test
	call	printf
	add	esp, 20
	ret
 
get_nums
	mov	eax, dword [esp+4]
	mov	ecx, dword [esp+8]
	ret
 
	section .data
 
_a	dd	11
_b	dd	3
 
	section .rodata
_test	db	'%08x %s %08x = %08x', 10, 0
_testn	db	'%08s %s %08x = %08x', 10, 0 
_opand	db	'and', 0
_opor	db	'or ', 0
_opxor	db	'xor', 0
_opshl	db	'shl', 0
_opshr	db	'shr', 0
_opror	db	'ror', 0
_oprol	db	'rol', 0
_opnot	db	'not', 0
_opsal	db	'sal', 0
_opsar	db	'sar', 0
_null	db 	0
 
	end