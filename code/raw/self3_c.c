char s[] = {
0x6D, 0x61, 0x69, 0x6E, 0x28, 0x29, 0x20, 0x7B, 
0x0D, 0x0A, 0x69, 0x6E, 0x74, 0x20, 0x69, 0x3B, 
0x0D, 0x0A, 0x09, 0x70, 0x72, 0x69, 0x6E, 0x74, 
0x66, 0x28, 0x22, 0x63, 0x68, 0x61, 0x72, 0x20, 
0x73, 0x5B, 0x5D, 0x20, 0x3D, 0x20, 0x7B, 0x22, 
0x29, 0x3B, 0x0D, 0x0A, 0x09, 0x66, 0x6F, 0x72, 
0x28, 0x69, 0x3D, 0x30, 0x3B, 0x73, 0x5B, 0x69, 
0x5D, 0x21, 0x3D, 0x27, 0x5C, 0x30, 0x27, 0x3B, 
0x69, 0x2B, 0x2B, 0x29, 0x20, 0x7B, 0x0D, 0x0A, 
0x09, 0x09, 0x69, 0x66, 0x28, 0x20, 0x28, 0x69, 
0x26, 0x37, 0x29, 0x3D, 0x3D, 0x30, 0x20, 0x29, 
0x20, 0x70, 0x72, 0x69, 0x6E, 0x74, 0x66, 0x28, 
0x22, 0x5C, 0x6E, 0x22, 0x29, 0x3B, 0x0D, 0x0A, 
0x09, 0x09, 0x70, 0x72, 0x69, 0x6E, 0x74, 0x66, 
0x28, 0x22, 0x30, 0x78, 0x25, 0x30, 0x32, 0x58, 
0x2C, 0x20, 0x22, 0x2C, 0x73, 0x5B, 0x69, 0x5D, 
0x29, 0x3B, 0x0D, 0x0A, 0x09, 0x7D, 0x0D, 0x0A, 
0x09, 0x70, 0x72, 0x69, 0x6E, 0x74, 0x66, 0x28, 
0x22, 0x30, 0x5C, 0x6E, 0x7D, 0x3B, 0x5C, 0x6E, 
0x25, 0x73, 0x5C, 0x6E, 0x22, 0x2C, 0x73, 0x29, 
0x3B, 0x0D, 0x0A, 0x7D, 0
};
main() {
int i;
	printf("char s[] = {");
	for(i=0;s[i]!='\0';i++) {
		if( (i&7)==0 ) printf("\n");
		printf("0x%02X, ",s[i]);
	}
	printf("0\n};\n%s\n",s);
}

