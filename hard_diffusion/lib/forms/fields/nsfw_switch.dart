import 'package:flutter/material.dart';

class NSFWSwitch extends StatefulWidget {
  const NSFWSwitch({super.key});

  @override
  State<NSFWSwitch> createState() => _NSFWSwitchState();
}

class _NSFWSwitchState extends State<NSFWSwitch> {
  bool light = true;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Text('NSFW:'),
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
