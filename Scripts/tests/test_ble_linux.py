# Standard modules
import random
import dbus
try:
    from gi.repository import GObject
except ImportError:
    import gobject as GObject

# Bluezero modules
from bluezero import constants
from bluezero import adapter
from bluezero import advertisement
from bluezero import localGATT
from bluezero import GATT

# constants
TEXT_SRVC = '6E408888-B5A3-F393-E0A9-E50E24DCCA9E'
TEXT_TX_CHRC = '6E409999-B5A3-F393-E0A9-E50E24DCCA9E'


def get_txt_value():
    text_values = ['Random words', 'Other stuff', 'Test value 2', 'Another sentence']
    return random.choice(text_values)


class TxChrc(localGATT.Characteristic):
    def __init__(self, service):
        localGATT.Characteristic.__init__(self,
                                          1,
                                          TEXT_TX_CHRC,
                                          service,
                                          [get_txt_value()],
                                          False,
                                          ['read', 'notify'])

    def value_cb(self):
        reading = get_txt_value()

        self.props[constants.GATT_CHRC_IFACE]['Value'] = reading.encode('utf-8')

        self.PropertiesChanged(constants.GATT_CHRC_IFACE,
                               {'Value': dbus.ByteArray(self.props[constants.GATT_CHRC_IFACE]['Value'])},
                               [])
        return self.props[constants.GATT_CHRC_IFACE]['Notifying']

    def _update_value(self):
        if not self.props[constants.GATT_CHRC_IFACE]['Notifying']:
            return

        print('Starting timer event')
        GObject.timeout_add(500, self.value_cb)

    def ReadValue(self, options):
        reading = get_txt_value()
        self.props[constants.GATT_CHRC_IFACE]['Value'] = reading.encode('utf-8')
        print('Value from ReadValue: {}'.format(reading))
        return dbus.ByteArray(self.props[constants.GATT_CHRC_IFACE]['Value'])

    def StartNotify(self):
        if self.props[constants.GATT_CHRC_IFACE]['Notifying']:
            print('Already notifying, nothing to do')
            return
        print('Notifying on')
        self.props[constants.GATT_CHRC_IFACE]['Notifying'] = True
        self._update_value()

    def StopNotify(self):
        if not self.props[constants.GATT_CHRC_IFACE]['Notifying']:
            print('Not notifying, nothing to do')
            return

        print('Notifying off')
        self.props[constants.GATT_CHRC_IFACE]['Notifying'] = False
        self._update_value()


class ble:
    def __init__(self):
        self.bus = dbus.SystemBus()
        self.app = localGATT.Application()
        self.srv = localGATT.Service(1, TEXT_SRVC, True)

        self.charc = TxChrc(self.srv)
        self.charc.service = self.srv.path

        self.app.add_managed_object(self.srv)
        self.app.add_managed_object(self.charc)

        self.srv_mng = GATT.GattManager(adapter.list_adapters()[0])
        self.srv_mng.register_application(self.app, {})

        self.dongle = adapter.Adapter(adapter.list_adapters()[0])
        advert = advertisement.Advertisement(1, 'peripheral')

        advert.service_UUIDs = [TEXT_SRVC]
        if not self.dongle.powered:
            self.dongle.powered = True
        ad_manager = advertisement.AdvertisingManager(self.dongle.address)
        ad_manager.register_advertisement(advert, {})

    def add_call_back(self, callback):
        self.charc.PropertiesChanged = callback

    def start_bt(self):
        self.app.start()


if __name__ == '__main__':
    pi_txt_service = ble()
    pi_txt_service.start_bt()