**Purpose of the Script:**
================

The purpose of this script is to demonstrate the process of creating a new network namespace, assigning two network devices to that namespace, and configuring their IP addresses. This is done to illustrate how network namespaces can be used to isolate and manage network resources within a Linux environment. The script then performs a ping test to verify the connectivity between the devices in separate namespaces.

**Step-by-Step Explanation:**
================

1. **Define Useful Variables**: The script begins by defining several variables. `ns1` serves as the name for the new network namespace, while `eth1` and `eth2` represent the names of the network devices to be utilized.

2. **Create a New Network Namespace**: Using the command `sudo ip netns add "$netns"`, the script creates a new network namespace named `$netns`.

3. **Bring Devices Down in Default Namespace**: The script employs `sudo ip link set dev "$dev1" down` and `sudo ip link set dev "$dev2" down` to deactivate the devices in the default namespace.

4. **Add a Device to the New Namespace**: One of the devices, `$dev1`, is moved to the new namespace using the command `sudo ip link set dev "$dev1" netns "$netns"`.

5. **Assign IP Addresses**: IP addresses are assigned to the devices. The script uses `sudo ip address add 192.168.1.5/24 dev "$dev2"` to assign an IP address to `$dev2`. Additionally, it assigns an IP address to `$dev1` within the new namespace using `sudo ip netns exec "$netns" ip address add 192.168.1.6/24 dev "$dev1"`.

6. **Confirm Device Locations and IP Addresses**: Device locations and their corresponding IP addresses are verified using the commands `sudo ip address show` and `sudo ip netns exec "$netns" ip address show`.

7. **Physically Connect the Two Interfaces**: If not previously connected, the script advises to physically connect the two network interfaces.

8. **Bring the Interfaces Up**: To activate the interfaces, the script employs commands like `sudo ip link set dev "$dev2" up`, `sudo ip netns exec "$netns" ip link set dev "$dev1" up`, and `sudo ip netns exec "$netns" ip link set dev lo up`. This includes bringing up the loopback device in the new namespace.

9. **Check Routes**: Routes are examined using `sudo ip route show` and `sudo ip netns exec "$netns" ip route show` to ensure proper configuration.

10. **Ping One Way or the Other**: The script performs ping tests with commands such as `$ping 192.168.1.6` and `ip netns exec "$netns" ping 192.168.1.5` to validate the connectivity between the devices located in separate namespaces.

11. 	1. Define useful variables. Here ns1 is an arbitrary name for a namespace; eth1 and eth2 are the devices you want to use.
    ```bash
$netns=ns1
$dev1=ens785f1
$dev2=ens786f1np1
	2. Create a new network namespace.
$sudo ip netns add "$netns"
	3. Bring the devices down in the default namespace.
$sudo ip link set dev "$dev1" down
$sudo ip link set dev "$dev2" down
	4. Add one of the devices to the new namespace (it will disappear from the default namespace). Here I choose to move $dev2 to the new namespace.
$sudo ip link set dev "$dev1" netns "$netns"
	5. Assign IP addresses.
                   $sudo ip address add 192.168.1.5/24 dev "$dev2"
	$sudo ip netns exec "$netns" ip address add 192.168.1.6/24 dev "$dev1"
	6. Confirm the two devices are where they should be, with the right IP addresses.
                      $sudo ip address show
$sudo ip netns exec "$netns" ip address show
# examine output
	7. Physically connect the two interfaces with a cable, if not yet done.
	8. Bring the interfaces up. The namespace contains its own loopback device lo. I bring it up just in case, because in general programs may want to rely on it.
                      $sudo ip link set dev "$dev2" up
$sudo ip netns exec "$netns" ip link set dev "$dev1" up
$sudo ip netns exec "$netns" ip link set dev lo up
	9. Check routes.
	$sudo ip route show
prints (among other lines)
192.168.1.0/24 dev eth1 proto kernel scope link src 192.168.1.5
and this command
$sudo ip netns exec "$netns" ip route show
prints
192.168.1.0/24 dev eth2 proto kernel scope link src 192.168.1.6
	10. Ping one way or the other.
                       $ping 192.168.1.6
$ip netns exec "$netns" ping 192.168.1.5 
```
