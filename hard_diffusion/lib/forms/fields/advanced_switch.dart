import 'package:flutter/material.dart';

class AdvancedSwitch extends StatefulWidget {
  const AdvancedSwitch({super.key});

  @override
  State<AdvancedSwitch> createState() => _AdvancedSwitchState();
}

class _AdvancedSwitchState extends State<AdvancedSwitch> {
  bool light = true;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Text('Advanced:'),
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
