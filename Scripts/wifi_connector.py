#
# Created on Wed Jan 26 2022
# Author: Owen Yip
# Mail: me@owenyip.com
#
import os


class WifiConnector:

    # function to establish a new connection
    def createNewConnection(name, SSID, password):
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
    def connect(name, SSID):
        command = "netsh wlan connect name=\""+name + \
            "\" ssid=\""+SSID+"\" interface=Wi-Fi"
        os.system(command)

    # function to display avavilabe Wifi networks
    def displayAvailableNetworks():
        command = "netsh wlan show networks interface=Wi-Fi"
        os.system(command)

if __name__ == '__main__':
    wifi_connector = WifiConnector()
    # display available netwroks
    wifi_connector.displayAvailableNetworks()

    # input wifi name and password
    name = input("Name of Wi-Fi: ")
    password = input("Password: ")

    # establish new connection
    wifi_connector.createNewConnection(name, name, password)

    # connect to the wifi network
    wifi_connector.connect(name, name)
    print("If you aren't connected to this network, try connecting with the correct password!")

