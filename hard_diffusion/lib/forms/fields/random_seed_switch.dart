import 'package:flutter/material.dart';

class RandomSeedSwitch extends StatefulWidget {
  const RandomSeedSwitch({super.key});

  @override
  State<RandomSeedSwitch> createState() => _RandomSeedSwitchState();
}

class _RandomSeedSwitchState extends State<RandomSeedSwitch> {
  bool light = true;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Text('Random:'),
          Switch(
            // This bool value toggles the switch.
            value: light,
            activeColor: Colors.orange,
            onChanged: (bool value) {
              // This is called when the user toggles the switch.
              setState(() {
                light = value;
              });
            },
          ),
        ],
      ),
    );
  }
}
