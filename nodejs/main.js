var bleno = require('bleno-mac');

var BlenoPrimaryService = bleno.PrimaryService;

var EchoCharacteristic = require('./characteristic');

var SmartWalker_UUID = '4df03521-6912-431d-89c7-5b3b63a648be';
var SetWifi_ControlPoint = 'b7e9e754-1e59-42d4-810f-16ed47244b50';

console.log('bleno - echo');

bleno.on('stateChange', function(state) {
  console.log('on -> stateChange: ' + state);

  if (state === 'poweredOn') {
    bleno.startAdvertising('echo', [SmartWalker_UUID]);
  } else {
    bleno.stopAdvertising();
  }
});

bleno.on('advertisingStart', function(error) {
  console.log('on -> advertisingStart: ' + (error ? 'error ' + error : 'success'));

  if (!error) {
    bleno.setServices([
      new BlenoPrimaryService({
        uuid: SmartWalker_UUID,
        characteristics: [
          new EchoCharacteristic()
        ]
      })
    ]);
  }
});