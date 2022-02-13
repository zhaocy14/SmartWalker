#
# Created on Wed Jan 26 2022
# Author: Owen Yip
# Mail: me@owenyip.com
#
import wifi

class WifiConnector:
    def Search(self):
        wifilist = []

        cells = wifi.Cell.all('wlan0')

        for cell in cells:
            wifilist.append(cell)

        return wifilist


    def FindFromSearchList(self, ssid):
        wifilist = self.Search()

        for cell in wifilist:
            if cell.ssid == ssid:
                return cell

        return False


    def FindFromSavedList(self, ssid):
        cell = wifi.Scheme.find('wlan0', ssid)

        if cell:
            return cell

        return False


    def Connect(self, ssid, password=None, hidden=False):
        if not hidden:
            cell = self.FindFromSearchList(ssid)

            if cell:
                savedcell = self.FindFromSavedList(ssid)

                # Already Saved from Setting
                if savedcell:
                    savedcell.activate()
                    return cell

                # First time to conenct
                else:
                    if cell.encrypted:
                        if password:
                            scheme = self.Add(cell, password)

                            try:
                                scheme.activate()

                            # Wrong Password
                            except wifi.exceptions.ConnectionError:
                                self.Delete(ssid)
                                return False

                            return cell
                        else:
                            return False
                    else:
                        scheme = self.Add(cell)

                        try:
                            scheme.activate()
                        except wifi.exceptions.ConnectionError:
                            self.Delete(ssid)
                            return False

                        return cell
        
        # Todo: Hidden network handling
        else:
            print('This is a hidden network')
        
        return False


    def Add(self, cell, password=None):
        if not cell:
            return False

        scheme = wifi.Scheme.for_cell('wlan0', cell.ssid, cell, password)
        scheme.save()
        return scheme


    def Delete(self, ssid):
        if not ssid:
            return False

        cell = self.FindFromSavedList(ssid)

        if cell:
            cell.delete()
            return True

        return False


if __name__ == '__main__':
    wifi_connector = WifiConnector()
    # Search WiFi and return WiFi list
    print(wifi_connector.Search())

    # Connect WiFi with password & without password
    print(wifi_connector.Connect('OpenWiFi'))
    print(wifi_connector.Connect('ClosedWiFi', 'password'))

    # Delete WiFi from auto connect list
    print(wifi_connector.Delete('DeleteWiFi'))