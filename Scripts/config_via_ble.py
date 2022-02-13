#
# Created on Tue Jan 25 2022
# Author: Owen Yip
# Mail: me@owenyip.com
#
# Standard modules

"""Installation
1. sudo apt-get install libffi-dev
2. sudo apt-get install bluez-tools
3. sudo apt install build-essential libdbus-glib-1-dev libgirepository1.0-dev
4. sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
5. Insall pyenv
6. pip3 install pycairo
7. pip3 install PyGObject
8. pip3 install dbus-python
9. pip3 install bluezero

"""

import logging
import random
import json

# Bluezero modules
# from bluezero import async_tools
from bluezero import adapter
from bluezero import peripheral

from wifi_connector import WifiConnector

# constants
# Custom service uuid
SmartWalkerUUID = "4df03521-6912-431d-89c7-5b3b63a648be"
# https://www.bluetooth.com/specifications/assigned-numbers/
# Bluetooth SIG adopted UUID for Temperature characteristic
SetWifiControlPoint = "b7e9e754-1e59-42d4-810f-16ed47244b50"
# Bluetooth SIG adopted UUID for Characteristic Presentation Format
WIFI_FMT_DSCP = '2904'


class BleManager:
    def __init__(self, address=None):
        if address is None:
            self.adpater_address = list(adapter.Adapter.available())[0].address
        else:
            self.adpater_address = address
    
    
    def read_value(self):
        """
        Example read callback. Value returned needs to a list of bytes/integers
        in little endian format.

        This one does a mock reading CPU temperature callback.
        Return list of integer values.
        Bluetooth expects the values to be in little endian format and the
        temperature characteristic to be an sint16 (signed & 2 octets) and that
        is what dictates the values to be used in the int.to_bytes method call.

        :return: list of uint8 values
        """
        cpu_value = random.randrange(3200, 5310, 10) / 100
        return list(int(cpu_value * 100).to_bytes(2,
                                                byteorder='little', signed=True))


    def wificonfig_notify_value(self, characteristic):
        """
        Example of callback to send notifications

        :param characteristic:
        :return: boolean to indicate if timer should continue
        """
        # read/calculate new value.
        new_value = self.read_value()
        # Causes characteristic to be updated and send notification
        characteristic.set_value(new_value)
        # Return True to continue notifying. Return a False will stop notifications
        # Getting the value from the characteristic of if it is notifying
        return characteristic.is_notifying


    def wificonfig_write_value(self, value, options):
        if value:
            print("Received: ", bytes(value).decode('utf-8'))
            wifi_connector = WifiConnector()
            wifi_config = json.loads(bytes(value).decode('utf-8'))
            print("parse: ", wifi_config)
            result = wifi_connector.Connect(ssid=wifi_config["ssid"], password=wifi_config["password"]["value"])
            print("connect result: ", result)
            
        return True


    # def notify_callback(notifying, characteristic):
    #     """
    #     Noitificaton callback example. In this case used to start a timer event
    #     which calls the update callback ever 2 seconds

    #     :param notifying: boolean for start or stop of notifications
    #     :param characteristic: The python object for this characteristic
    #     """
    #     if notifying:
    #         async_tools.add_timer_seconds(2, update_value, characteristic)


    def start(self):
        """Creation of peripheral"""
        logger = logging.getLogger('localGATT')
        logger.setLevel(logging.DEBUG)
        # Example of the output from read_value
        print('Advertising SmartWalker BLE')
        # Create peripheral
        ble_configurator = peripheral.Peripheral(self.adapter_address,
                                            local_name='SmartWalker BLE Configurator')
        
        # Add service
        ble_configurator.add_service(srv_id=1, uuid=SmartWalkerUUID, primary=True)
        # Add characteristic
        ble_configurator.add_characteristic(srv_id=1, chr_id=1, uuid=SetWifiControlPoint,
                                    value=[], notifying=False,
                                    flags=['write', 'notify'],
                                    read_callback=None,
                                    write_callback=self.wificonfig_write_value,
                                    notify_callback=self.wificonfig_notify_value
                                    )
        # Add descriptor
        # ble_configurator.add_descriptor(srv_id=1, chr_id=1, dsc_id=1, uuid=WIFI_FMT_DSCP,
        #                            value=[0x0E, 0xFE, 0x2F, 0x27, 0x01, 0x00,
        #                                   0x00],
        #                            flags=['read'])
        # Publish peripheral and start event loop
        ble_configurator.publish()


if __name__ == '__main__':
    # Get the default adapter address and pass it to main
    # print(list(adapter.Adapter.available())[0].address)
    # main(list(adapter.Adapter.available())[0].address)
    ble_manager = BleManager()
