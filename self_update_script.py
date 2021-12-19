from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import re
from subprocess import call
from operator import eq
import os
import sys
import stat
import requests
from requests.exceptions import HTTPError
import time

status_codes = {
    200: {
        "result": True,
        "message": "You already have the latest version."
    },
    201: {
        "result": True,
        "message": "New version installed"
    },
    202: {
        "result": True,
        "message": "Failed to update, rollbacked to the old version"
    },
    300: {
        "result": False,
        "message": "Walker state not allowed"
    },
    301: {
        "result": False,
        "message": "Failed to start the program"
    },
    400: {
        "result": False,
        "message": "Unable to parse version datad"
    },
    401: {
        "result": False,
        "message": "Unable to download the binary"
    },
    500: {
        "result": False,
        "message": "Cannot update -- unable to write to the program folder"
    },
    501: {
        "result": False,
        "message": "Cannot update -- unable to rename the program file"
    }
}

def update(version_url="https://owenyip.com/smartwalker.json", app_path="/Users/owen/Documents/APP/Python/SmartWalker/dist/testupdate", force_update=False):
    """
    Attempts to download the update url in order to find if an update is needed.
    If an update is needed, the current script is backed up and the update is
    saved in its place.
    """
    def compare_versions(vA, vB):
        """
        Compares two version number strings
        @param vA: first version string to compare
        @param vB: second version string to compare
        @author <a href="http_stream://sebthom.de/136-comparing-version-numbers-in-jython-pytho/">Sebastian Thomschke</a>
        @return negative if vA < vB, zero if vA == vB, positive if vA > vB.
        """
        if vA == vB:
            return 0

        def num(s):
            return int(s)

        vA_list = re.findall('\d+|\w+', vA)
        vB_list = re.findall('\d+|\w+', vB)
        seqA = list(map(num, vA_list))
        seqB = list(map(num, vB_list))

        # this is to ensure that 1.0 == 1.0.0 in cmp(..)
        lenA, lenB = len(seqA), len(seqB)
        for i in range(lenA, lenB):
            seqA += (0,)
        for i in range(lenB, lenA):
            seqB += (0,)
        print(seqA, seqB)
        print(seqA == seqB)
        # rc = (seqA == seqB)
        return seqA == seqB

    # get current version
    current_version = "1.0.0"

    # get the version number from url
    jsonResponse = None
    dl_url = ""
    update_version = None
    dl_path = app_path + ".new"
    backup_path = app_path + ".old"
    try:
        response = requests.get(version_url)
        response.raise_for_status()
        # access JSOn content
        jsonResponse = response.json()
        # print("Entire JSON response")
        # print(jsonResponse)
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

    if jsonResponse is not None:
        update_version = jsonResponse["version"]
        dl_url = jsonResponse["downloadUrl"]

    if not update_version:
        print("Unable to parse version data")
        return 400

    if force_update:
        print("Forcing update, downloading version %s..."
              % update_version)
    else:
        cmp_result = compare_versions(current_version, update_version)
        if not cmp_result:
            print("Newer version %s available, downloading..." % update_version)
        else:
            print("You already have the latest version.")
            return 200
        
    download_result = download(app_path, dl_path, dl_url)
    if download_result != True:
        return download_result
        
    """Check walker idle state and whether in power station
    if yes, need to block other actions
    if no, need to retry
    """
    retry_count = 0
    while not is_walker_idle() and not is_walker_in_power_station():
        retry_count += 1
        time.sleep(10 * 60) # sleep for 60 seconds
        if retry_count >= 6: # max retry 6 times
            return 300
    
    # Todo: Stop the walker program
    stop_walker_program()
    # Todo: Change walker state to "Updating"
    change_walker_state("Updating")
    
    # Replace the old program with the new
    replace_program_result = replace_program(app_path, dl_path, backup_path)
    if replace_program_result != 201:
        return replace_program_result
    
    def start_program_successful():
        retry_count = 0
        while not start_walker_program():
            retry_count += 1
            time.sleep(30) # sleep for 30 seconds
            if retry_count >= 20: # max retry 20 times
                return False
        return True
    
    if not start_program_successful():
        # Rollback to the old version
        replace_program_result = replace_program(app_path, backup_path, dl_path)
        if replace_program_result != 201:
            return replace_program_result
    else:
        return 201

    if not start_program_successful():
        return 301
    else:
        return 202


def download(source_path, target_path):
    if not os.access(source_path, os.W_OK):
        print("Cannot update -- unable to write to %s" % source_path)
        return 500
        
    req = Request(dl_url)
    try:
        http_stream = urlopen(req)
        dl_file = open(target_path, 'wb')
        total_size = None
        bytes_so_far = 0
        chunk_size = 8192
        try:
            total_size = int(http_stream.info().getheader(
                'Content-Length').strip())
        except:
            # The header is improper or missing Content-Length, just download
            dl_file.write(http_stream.read())

        while total_size:
            chunk = http_stream.read(chunk_size)
            dl_file.write(chunk)
            bytes_so_far += len(chunk)

            if not chunk:
                break

            percent = float(bytes_so_far) / total_size
            percent = round(percent*100, 2)
            sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" %
                             (bytes_so_far, total_size, percent))

            if bytes_so_far >= total_size:
                sys.stdout.write('\n')

        http_stream.close()
        dl_file.close()
    except HTTPError as e:
        # do something
        print('Error code: ', e.code)
        return 401
    except URLError as e:
        # do something
        print('Reason: ', e.reason)
        return 401
    else:
        # do something
        print('Download finished!')
    return True


# Todo: Check walker idle state
def is_walker_idle():
    return True


# Todo: Check walker in power station
def is_walker_in_power_station():
    return True


# Todo: Check walker in power station
def change_walker_state(state=""):
    return True


# Todo: Check walker in power station
def stop_walker_program():
    return True


# Todo: Start the walker program
def start_walker_program():
    return True


def replace_program(source_path, target_path, backup_path):
    try:
        os.rename(source_path, backup_path)
    except OSError:
        print("Unable to rename %s to %s"
              % (source_path, backup_path))
        return 501

    try:
        os.rename(target_path, app_path)
    except OSError:
        print("Unable to rename %s to %s"
              % (target_path, app_path))
        return 501

    try:
        import shutil
        shutil.copymode(backup_path, app_path)
    except:
        os.chmod(app_path, 0o755)

    print("New version installed as %s" % app_path)
    print("(previous version backed up to %s)" % (backup_path))
    return 201


if __name__ == "__main__":
    version_url = "https://owenyip.com/smartwalker.json"
    dl_url = "https://owenyip.com/smartwalker.app"
    app_path = "/Users/owen/Documents/APP/Python/SmartWalker/dist/testupdate"

    result_code = update()
    if result_code:
        print(status_codes[result_code])
