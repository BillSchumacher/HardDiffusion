import 'package:flutter/material.dart';

class RandomSeedSwitch extends StatefulWidget {
  const RandomSeedSwitch(
      {super.key, required this.value, required this.setValue});

  final bool value;
  final Function(bool) setValue;
  @override
  State<RandomSeedSwitch> createState() =>
      _RandomSeedSwitchState(value: value, setValue: setValue);
}

class _RandomSeedSwitchState extends State<RandomSeedSwitch> {
  _RandomSeedSwitchState({required this.value, required this.setValue});

  bool value;
  Function setValue;
  bool light = true;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text('Random Seed:'),
        Switch(
          // This bool value toggles the switch.
          value: light,
          activeColor: Colors.orange,
          onChanged: (bool newValue) {
            // This is called when the user toggles the switch.
            setState(() {
              light = newValue;
            });
            setValue(light);
          },
        ),
      ],
    );
  }
}
