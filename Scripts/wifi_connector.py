#
# Created on Wed Jan 26 2022
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os
import subprocess
import nmcli

class WifiConnector:

    # function to establish a new connection
    def createNewConnection(self, name, SSID, password):
        config = """<?xml version=\"1.0\"?>
    <WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
        <name>"""+name+"""</name>
        <SSIDConfig>
            <SSID>
                <name>"""+SSID+"""</name>
            </SSID>
        </SSIDConfig>
        <connectionType>ESS</connectionType>
        <connectionMode>auto</connectionMode>
        <MSM>
            <security>
                <authEncryption>
                    <authentication>WPA2PSK</authentication>
                    <encryption>AES</encryption>
                    <useOneX>false</useOneX>
                </authEncryption>
                <sharedKey>
                    <keyType>passPhrase</keyType>
                    <protected>false</protected>
                    <keyMaterial>"""+password+"""</keyMaterial>
                </sharedKey>
            </security>
        </MSM>
    </WLANProfile>"""
        command = "netsh wlan add profile filename=\""+name+".xml\""+" interface=Wi-Fi"
        with open(name+".xml", 'w') as file:
            file.write(config)
        os.system(command)

    # function to connect to a network
    def connect(self, name, SSID):
        command = "netsh wlan connect name=\""+name + \
            "\" ssid=\""+SSID+"\" interface=Wi-Fi"
        os.system(command)

    # function to display avavilabe Wifi networks
    def displayAvailableNetworks(self):
        command = "netsh wlan show networks interface=Wi-Fi"
        os.system(command)

if __name__ == '__main__':
    try:
        # print(nmcli.connection()) # Saved connections
        # print(nmcli.device()) # Get all network devices
        # print(nmcli.device.wifi()) # Get all available wifis
        # print(nmcli.general()) # Get current wifi connection state General(state=<NetworkManagerState.CONNECTED_GLOBAL: 'connected'>, connectivity=<NetworkConnectivity.FULL: 'full'>, wifi_hw=True, wifi=True, wwan_hw=True, wwan=True)
        connection_status = nmcli.general().to_json()
        print(connection_status['state'])

        nmcli.device.wifi_connect('AP1', 'passphrase')
        # nmcli.connection.modify('AP1', {
        #         'ipv4.addresses': '192.168.1.1/24',
        #         'ipv4.gateway': '192.168.1.255',
        #         'ipv4.method': 'manual'
        #     })
        # nmcli.connection.down('AP1')
        # nmcli.connection.up('AP1')
        # nmcli.connection.delete('AP1')
        
    except Exception as e:
        print('catch:', e)

