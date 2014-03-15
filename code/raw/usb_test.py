def main():
    import usb.core
    import usb.util
    import usb.backend
    import sys
    
    VENDOR_ID = 0x046d
    PRODUCT_ID = 0xc214

    device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    print device

    for i in range(10):
        print device.read(0x1d100000, 10)
main()