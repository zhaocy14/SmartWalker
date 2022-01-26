#
# Created on Wed Jan 26 2022
# Author: Owen Yip
# Mail: me@owenyip.com
#
import wifi


def Search():
    wifilist = []

    cells = wifi.Cell.all('wlan0')

    for cell in cells:
        wifilist.append(cell)

    return wifilist


def FindFromSearchList(ssid):
    wifilist = Search()

    for cell in wifilist:
        if cell.ssid == ssid:
            return cell

    return False


def FindFromSavedList(ssid):
    cell = wifi.Scheme.find('wlan0', ssid)

    if cell:
        return cell

    return False


def Connect(ssid, password=None, hidden=False):
    if not hidden:
        cell = FindFromSearchList(ssid)

        if cell:
            savedcell = FindFromSavedList(ssid)

            # Already Saved from Setting
            if savedcell:
                savedcell.activate()
                return cell

            # First time to conenct
            else:
                if cell.encrypted:
                    if password:
                        scheme = Add(cell, password)

                        try:
                            scheme.activate()

                        # Wrong Password
                        except wifi.exceptions.ConnectionError:
                            Delete(ssid)
                            return False

                        return cell
                    else:
                        return False
                else:
                    scheme = Add(cell)

                    try:
                        scheme.activate()
                    except wifi.exceptions.ConnectionError:
                        Delete(ssid)
                        return False

                    return cell
    
    else:
        print('This is a hidden network')
    
    return False


def Add(cell, password=None):
    if not cell:
        return False

    scheme = wifi.Scheme.for_cell('wlan0', cell.ssid, cell, password)
    scheme.save()
    return scheme


def Delete(ssid):
    if not ssid:
        return False

    cell = FindFromSavedList(ssid)

    if cell:
        cell.delete()
        return True

    return False


if __name__ == '__main__':
    # Search WiFi and return WiFi list
    print(Search())

    # Connect WiFi with password & without password
    print(Connect('OpenWiFi'))
    print(Connect('ClosedWiFi', 'password'))

    # Delete WiFi from auto connect list
    print(Delete('DeleteWiFi'))