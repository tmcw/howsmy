#!/bin/bash
for i in {1..500}
do
  curl 'https://wmq.etimspayments.com/pbw/CaptchaServlet.doh?657' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:74.0) Gecko/20100101 Firefox/74.0' -H 'Accept: image/webp,*/*' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Referer: https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp' -H 'Cookie: visid_incap_391913=0sOd8ay3SYufCZtvo23wt+QQgV4AAAAAQUIPAAAAAAB4GkJRicELtm06nFo4ROpf; incap_ses_442_391913=h6gCPUl+xEjchfc0ZU4iBor/iF4AAAAA6+ii3sdlCY+e6ZZz/EO4FA==; JSESSIONID=0000waL307fZk_6ksmByPBmWC5M:17bhd4lo9' -H 'TE: Trailers' > "unlabeled-color/${i}.png"
  sleep 1
done
