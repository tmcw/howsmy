#!/bin/bash
for i in {1..3000}
do
  curl 'https://wmq.etimspayments.com/pbw/CaptchaServlet.doh' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:74.0) Gecko/20100101 Firefox/74.0' -H 'Accept: image/webp,*/*' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Referer: https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp' -H 'Cookie: visid_incap_391913=XdeNMcYJTPy3Xg4ijcUcIdUTil4AAAAAQUIPAAAAAADEtemCWK1Z64t2Fsl8ZA8u; incap_ses_458_391913=60B/fPq+Hn6rSivSICVbBrt1i14AAAAA36LQHC8cQEb6ZedSMw5CnQ==; JSESSIONID=0000HvKKVFVuJUlnpQ4L1p_qXO4:17bhd4oaa' -H 'TE: Trailers' > "trainset_captchas/${i}.png"
  sleep 1
done
