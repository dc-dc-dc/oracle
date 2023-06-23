import unittest, subprocess, json

class TestCLInfo(unittest.TestCase):
    def test_opencl_clinfo(self):
        clinfo_raw = subprocess.check_output(['clinfo', "--raw"])
        lines = [x.strip() for x in clinfo_raw.decode("utf-8").split("\n")]
        # TODO: test multiple platforms
        numplatforms = 0
        clinfo_res = dict()
        for line in lines:
            if line.startswith("#"):
                numplatforms = int(line.split(" ")[-1])
            elif line.startswith("["):
                device = line[:5].split("/")[-1]
                if device == "*":
                    continue
                line = line[6:].strip()
                name = line.split(" ")[0]
                value = line[len(name):].strip()
                if "devices" not in clinfo_res: clinfo_res["devices"] = dict()
                if device not in clinfo_res["devices"]: clinfo_res["devices"][device] = dict()
                if name != "" and value != "": clinfo_res["devices"][device][name] = value
            else:
                name = line.split(" ")[0]
                value = line[len(name):].strip()
                if name != "" and value != "": clinfo_res[name] = value

        oracle_res = json.loads(subprocess.check_output(['./oracle', '--opencl']).decode("utf-8"))
        def get_platform_oracle(platform: str): # -> dict
            for _,p in oracle_res.items():
                if p["name"] == platform:
                    return p
            return None

        platform = None
        for k,v in clinfo_res.items():
            if k == "CL_PLATFORM_NAME":
                platform = get_platform_oracle(v)
                break

        if platform == None:
            print("ERROR platform not found in oracle")    

        for k,v in clinfo_res.items():
            if k == "CL_PLATFORM_NAME": assert platform["name"] == v
            elif k == "CL_PLATFORM_VENDOR": assert platform["vendor"] == v
            elif k == "CL_PLATFORM_VERSION": assert platform["version"] == v
            elif k == "CL_PLATFORM_PROFILE": assert platform["profile"] == v
            elif k == "CL_PLATFORM_EXTENSIONS": self.assertListEqual(platform["extensions"], v.split(" "))
            elif k == "devices":
                for dk,dv in v.items():
                    print(dk)
                    pass
            else:
                print(f"ERROR: Key {k} not in oracle")

if __name__ == '__main__':
    unittest.main()